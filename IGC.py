import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from scipy import stats
import time
import matplotlib.pyplot as plt

def simulate_data_random(ch, conn, noise_level, order=5, nonlinear=False, fixed_weight=False, sn=10000):
    """
    simulate the data
    ch is the number of features
    conn contains intended connectivity by linear modelling
    conn is a list of two nodes (ex: ([1, 2], [3, 4])) 1->2 & 3->4
    order is the number of effective samples
    noise_level is the level of noise
    sn=number of samples
    """
    x=torch.randn(sn, ch)
    for c in conn:
        x[order:,c[1]]=torch.zeros(sn-order)
    noise=torch.randn(x.shape[0], ch)
    if fixed_weight:
        weight=torch.rand(order)*4-2
        weights=weight
    else:
        weights=[]
    for nodes in conn:
        if fixed_weight is False:
          weight=torch.rand(order)*4-2
          weights.append(weight)
        """
        for i in range(order, sn):
          if nonlinear:
            x[i,nodes[1]]+=torch.sum([wei*x[i-a-1,nodes[0]]**(order-a) for a, wei in enumerate(weight)])
          else:
            x[i,nodes[1]]+=torch.sum(weight*x[i-order:i,nodes[0]].flip(0))
        """
        x[order:,nodes[1]]+=F.conv1d(x[:,nodes[0]].reshape(1,1,-1), weight.flip(-1).reshape(1,1,-1))[0,0,:-1]
    return x+noise*noise_level, weights

def simulate_data_random_with_weights(ch, conn, noise_level, weights, sn=10000):
    """
    simulate the data
    ch is the number of features
    conn contains intended connectivity by linear modelling
    conn is a list of two nodes (ex: ([1, 2], [3, 4])) 1->2 & 3->4
    order is the number of effective samples
    noise_level is the level of noise
    sn=number of samples
    """
    x=torch.randn(sn, ch)
    order=weights[0].shape[0]
    for c in conn:
        x[order:,c[1]]=torch.zeros(sn-order)
    noise=torch.randn(x.shape[0], ch)
    for nodes, weight in zip(conn, weights):
        for i in range(order, sn):
            x[i,nodes[1]]+=torch.sum(weight*x[i-order:i,nodes[0]].flip(0))
    return x+noise*noise_level

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=True):
    data=[]
    labels=[]
    start_index=start_index+history_size
    if end_index is None:
        end_index=len(dataset)-target_size
    for i in range(start_index, end_index):
        indices=range(i-history_size, i, step)
        data.append(dataset[indices])
        if single_step:
            labels.append(target[i])
        else:
            labels.append(target[i:i+target_size])
    if isinstance(dataset, torch.Tensor):
        return torch.stack(data,0), torch.stack(labels,0)
    else:
        return np.array(data), np.array(labels)

def get_gradients(model, inputs, verbose=True):
    model.train(True)
    for out_n in model.parameters():
        model_outsize=out_n.shape
    model_outsize=model_outsize[0]
    device=next(model.parameters()).device
    grads=torch.zeros((len(inputs), inputs[0].shape[-2], inputs[0].shape[-1], model_outsize))
    for interp_num, interp_input in enumerate(inputs):
        y=model(interp_input.to(device))
        for i in range(y.shape[-1]):
            y_part=y.gather(1, torch.ones([1,1],dtype=torch.int64,device=device)*i)
            interp_input.grad = None
            y_part.backward(retain_graph=True)
            #igs[:,:,i]+=interp_input.grad.squeeze()
            grads[interp_num,:,:,i]=interp_input.grad.squeeze()
        if verbose: print('.', end='')
    print(f'{len(inputs)} gradients were collected')

    return grads

def get_gradients_VF(model, inputs, verbose=True):
    model.train(True)
    for out_n in model.parameters():
        model_outsize=out_n.shape
    model_outsize=model_outsize[0]
    device=next(model.parameters()).device
    #grads=torch.zeros((len(inputs), inputs[0].shape[-2], inputs[0].shape[-1], model_outsize))
    interp_input=torch.cat(inputs).float().requires_grad_(True)
    y=model(interp_input.to(device))
    # grads [batch, sequence, input, output]
    grads = torch.zeros([y.shape[0], interp_input.shape[1], interp_input.shape[2], model_outsize])
    # Compute gradients for all outputs at once
    for i in range(model_outsize):
        grad_outputs=torch.zeros_like(y)
        grad_outputs[:,i]=1.0
        grad = torch.autograd.grad(
            outputs=y,
            inputs=interp_input,
            grad_outputs=grad_outputs,  # Sum gradients for all output dimensions
            create_graph=False,
            retain_graph=True
        )[0]  # grads shape: (batch, sequence, input)
        grads[:,:,:,i]=grad # grads shape: (batch, sequence, input, output)
        if verbose and (i+1)%5==0: print('.',end='')
    return grads.to('cpu')

def get_integrated_gradients(model, x, baseline=None, num_grads=20, side=2, verbose=True):
    """
    igs = get_integrated_gradients(model, x, baseline=None, num_grads=100)
    model    : pytorch DNN model
    x        : reference input
    baseline : default (None) indicates 0
    num_grads: number of gradient steps from baseline to the reference input
    side     : if 1, only uses positive values (ex: images, min-max normalized signals)
               if 2, uses both of negative and positive values separately and merges both values
    """
    if baseline is None:
        baseline = torch.zeros(x.shape).float()
    else:
        baseline=baseline.float()
    
    # 1. Interpolation to reconstruct step inputs
    x=x.float()
    inputs1=[]
    inputs2=[]
    for step in range(1,num_grads+1):
        interpolated_signal1=(baseline+(step/num_grads)*(x-baseline)).unsqueeze(0).clone().detach()
        interpolated_signal1.requires_grad_(True)
        inputs1.append(interpolated_signal1)
        if side==2:
            interpolated_signal2=(baseline+(step/num_grads)*(-x-baseline)).unsqueeze(0).clone().detach()
            interpolated_signal2.requires_grad_(True)
            inputs2.append(interpolated_signal2)
    
    # 2. Remove random effects from the model
    n_model = copy.deepcopy(model)
    for child in n_model.children():
        if isinstance(child, nn.Dropout):
            child.p = 0
    
    # 3. Get gradients
    #grads1=get_gradients(n_model, inputs1, verbose=verbose)
    grads1=get_gradients_VF(n_model, inputs1, verbose=verbose)
    # Approximate the integral using the trapezoidal rule
    grads1 = (grads1[:-1] + grads1[1:]) / 2.0
    # 4. Calculate integrated gradients
    igs1 = (x - baseline).squeeze().unsqueeze(2).repeat(1,1,grads1.shape[-1]) * grads1.mean(0)

    if side==2:
        # For negative activity (Repeat 3~4)
        #grads2=get_gradients(n_model, inputs2, verbose=verbose)
        grads2=get_gradients_VF(n_model, inputs2, verbose=verbose)
        grads2 = (grads2[:-1] + grads2[1:]) / 2.0
        igs2 = (-x - baseline).squeeze().unsqueeze(2).repeat(1,1,grads2.shape[-1]) * grads2.mean(0)
    
        igs=(igs1-igs2)/2
    else:
        igs = igs1

    return igs

def get_integrated_gradients_separate(model, x, baseline=None, num_grads=20, side=2, verbose=True):
    """
    igs = get_integrated_gradients(model, x, baseline=None, num_grads=100)
    model    : pytorch DNN model
    x        : reference input
    baseline : default (None) indicates 0
    num_grads: number of gradient steps from baseline to the reference input
    side     : if 1, only uses positive values (ex: images, min-max normalized signals)
               if 2, uses both of negative and positive values separately and merges both values
    """
    if baseline is None:
        baseline = torch.zeros(x.shape).float()
    else:
        baseline=baseline.float()
    
    # 1. Interpolation to reconstruct step inputs
    x=x.float()
    M=x.shape[-1]
    inputs1=[]
    inputs2=[]
    for inp in range(M):
        x_p=torch.zeros_like(x)
        x_p[:,inp]=x[:,inp]
        for step in range(1,num_grads+1):
            interpolated_signal1=(baseline+(step/num_grads)*(x_p-baseline)).unsqueeze(0).clone().detach()
            interpolated_signal1.requires_grad_(True)
            inputs1.append(interpolated_signal1)
            if side==2:
                interpolated_signal2=(baseline+(step/num_grads)*(-x_p-baseline)).unsqueeze(0).clone().detach()
                interpolated_signal2.requires_grad_(True)
                inputs2.append(interpolated_signal2)
    
    # 2. Remove random effects from the model
    n_model = copy.deepcopy(model)
    for child in n_model.children():
        if isinstance(child, nn.Dropout):
            child.p = 0
    
    # 3. Get gradients
    #grads1=get_gradients(n_model, inputs1, verbose=verbose)
    grads1=get_gradients_VF(n_model, inputs1, verbose=verbose)
    # Gather corresponding inputs and outputs
    grads1 = grads1.view(M,num_grads,x.shape[-2],M,M).diagonal(dim1=0,dim2=3).transpose(-1,-2)
    # Approximate the integral using the trapezoidal rule
    grads1 = (grads1[:-1] + grads1[1:]) / 2.0
    # 4. Calculate integrated gradients
    igs1 = (x - baseline).squeeze().unsqueeze(2).repeat(1,1,grads1.shape[-1]) * grads1.mean(0)

    if side==2:
        # For negative activity (Repeat 3~4)
        #grads2=get_gradients(n_model, inputs2, verbose=verbose)
        grads2=get_gradients_VF(n_model, inputs2, verbose=verbose)
        grads2 = grads2.view(M,num_grads,x.shape[-2],M,M).diagonal(dim1=0,dim2=3).transpose(-1,-2)
        grads2 = (grads2[:-1] + grads2[1:]) / 2.0
        igs2 = (-x - baseline).squeeze().unsqueeze(2).repeat(1,1,grads2.shape[-1]) * grads2.mean(0)
    
        igs=(igs1-igs2)/2
    else:
        igs = igs1

    return igs

def ig_connectivity(x, model=None, order=20, num_grads=50, foi=torch.arange(0,100,5), sfreq=1000, stat='np', nrs=0, side=2, device=None):
    """
    Integrated Gradients connectivity
    igc, igc_f = ig_connectivity(x, model=None, order=20, foi=torch.arange(0,100,10), sfreq=1000)
        x     : 2D input tensor (N samples x M channels)
        model : give a torch prediction model (if default None, the function uses a basic LSTM model)
        order : order of a regression model (default=20)
        num_grads: number of gradient steps from baseline to input
        foi   : frequency of interest (default: 0~100)
        sfreq : sampling frequency
        stat  : statistics method. one of 'np' or 'pdc'
                'np' means non-parametric statistics with pseudo-distribution (default)
                'pdc' provides PDC and DTF normalization.
        nrs   : number of random signals (default is zero)
                It inserts additional random signals into 'x' so as to estimate random connectivity
        device: 'cuda' or 'cpu'
                If you don't provide anything, it tests whether CUDA is available.
        ----
        returns
        igc   : connectivity between channels. 
                Frequency-averaged representative connectivity (M,M)(columns cause rows)
        cl    : Confidence level
        igc_t : T-statistics
        igc_p : P-value
        igc_f : connectivity spectrum according to foi
        igc_o : Predicted connected weights
    """
    if isinstance(x, torch.Tensor)==False:
        print('Error: x should be a torch tensor')
        return None, None
    
    if device is None:
        device = ("cuda" if torch.cuda.is_available() else "cpu")
    # Build a predefined model and train
    if model is None:
        # Add random signals for statistical inference
        if (stat=='np' or stat=='NP'):
            if nrs==0: nrs=int(np.ceil(x.shape[-1]*.1))
            x = torch.cat([x, torch.randn(x.shape[0], nrs)], axis=1)
        # Normalization of input data
        for i in range(x.shape[1]):
            x[:,i]=(x[:,i]-torch.mean(x[:,i]))/torch.std(x[:,i])/3.0
        # Reconstruction of a dataset with (order) time steps (ex: 20)
        x_train, y_train = multivariate_data(x, x, 0, int(x.shape[0]), order, 1, 1, True)
        x_train = x_train.float()
        y_train = y_train.float()
        x_test, y_test = multivariate_data(x, x, int(x.shape[0]*0.8), x.shape[0], order, 1, 1, True)
        x_test = x_test.float()
        y_test = y_test.float()
        model = MVAR_biLSTM(x.shape[-1], [200, 200, 200], 1, x.shape[-1]).to(device)
        train_model(model, (x_train, y_train), (x_test, y_test), num_epochs=50, device=device)
        # Use a trained model
        # Validation of model fitting
        pred_y = evaluate_testset(model, (x_train,y_train))
        corr_train=torch.corrcoef(torch.concatenate(
            (pred_y.reshape(1,-1),y_train.reshape(1,-1)),axis=0))
        corr_train=corr_train.cpu().detach().numpy()
        # Transform R to T
        corr_t = corr_train[0,1]*np.sqrt(torch.numel(pred_y)-2)/np.sqrt(1-corr_train[0,1]**2)
        # Estimate T to P
        corr_p = 2*(1 - stats.t.cdf(np.abs(corr_t), torch.numel(pred_y)-2))
        print("Train accuracy (CC) : %5.3f  (p=%7.5f)" %(corr_train[0,1], corr_p))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    # Connectivity measure
    ref = torch.ones((order, x.shape[-1]), dtype=torch.float)
    igc = get_integrated_gradients(model, ref, num_grads=num_grads, side=side)
    igc_pdc, igc_dtf, igc_f = igs_spectrum(igc, foi, sfreq, mode=2)
    
    igc_o = igc.transpose(1,2)
    #igc = igc_f.mean(0)
    igc = igc_f.max(0)[0]
    if stat=='np' or stat=='NP':
        pseudo_dist=torch.cat([igc[-nrs:,:].reshape(-1), igc[:-nrs,-nrs:].reshape(-1)], axis=0)
        igc = igc[:-nrs,:-nrs]
        #igc_o = igc_o[:,:-nrs,:-nrs]
        igc_f = igc_f[:,:-nrs,:-nrs]
        cl=pseudo_dist.mean()+stats.t.ppf(
            0.01/igc.shape[0]**2/pseudo_dist.shape[0],pseudo_dist.shape[0]) * \
            pseudo_dist.std()/np.sqrt(pseudo_dist.shape[0])
        igc_t = (igc - pseudo_dist.mean())/torch.sqrt(pseudo_dist.var()/pseudo_dist.size(-1))
        igc_p = 2*(1 - stats.t.cdf(igc_t, pseudo_dist.size(-1)-1))*igc.shape[0]**2
        end.record()
        torch.cuda.synchronize()
        print('IGC computation time: %.3fs' %(start.elapsed_time(end)/1000))
        return igc, (cl, pseudo_dist), igc_t, igc_p, igc_f, igc_o, foi

    print('IGC computation time: %.3fs' %(start.elapsed_time(end)/1000))
    return igc_pdc, igc_dtf, igc_o, igc, igc_f, foi

def igs_spectrum(igs, foi=torch.arange(0,100,10), sfreq=100, mode=1):
    """
    Fourier transform of IG time steps
    igs_f = igs_spectrum(igs, mode=1)
        igs : integrated gradients from get_integrated_gradients() function
        foi : frequency of interest (default number = number of time steps)
        mode: 1 (default) -> Fourier transform only
                return igs_f
              2 -> PDC and DTF normalization
                return igs_pdc, igs_dtf, igs_f
              3 -> confidence level of mean igs_f
    """
    igs_f=torch.zeros((foi.shape[0],igs.shape[1],igs.shape[2]),dtype=torch.complex64)
    for f, frq in enumerate(foi):
        for i in range(igs.shape[0]):
            igs_f[f,:,:]+=igs[i,:,:].type(torch.complex64)*torch.exp(-1j*2*torch.pi*frq*i/sfreq)

    if mode == 2 and igs.shape[-2]==igs.shape[-1]:
        # In igs_f, rows indicate input features and columns are outputs
        # igs_f is needed to be transposed to represent "A column causes a row"
        M=igs.shape[-1]
        igs_pdc=torch.zeros((foi.shape[0],M,M), dtype=torch.float32)
        igs_dtf=torch.zeros((foi.shape[0],M,M), dtype=torch.float32)
        Af = torch.eye(M).repeat(igs_f.shape[0],1,1) - igs_f
        for f in range(foi.shape[0]):
            Aff=torch.abs(Af[f,:,:].T)
            igs_pdc[f,:,:] = (Aff/torch.sqrt(torch.diag(torch.matmul(torch.matmul(
                Aff.T,torch.eye(M)),Aff))).unsqueeze(0).repeat(M,1)).float()
            # Partial coherence
            #for row in range(M):
            #    for col in range(M):
            #        igs_pdc[f,row,col] = Aff[row,col]/torch.sqrt(Aff[row,row]*Aff[col,col])
            Hf = torch.abs(torch.linalg.inv(Aff))
            igs_dtf[f,:,:] = (Hf/torch.sqrt(torch.abs(torch.diag(torch.matmul(torch.matmul(
                Hf,torch.eye(M)),Hf.T)))).unsqueeze(1).repeat(1,M)).float()

        return igs_pdc, igs_dtf, torch.abs(igs_f.transpose(1,2))
    
    return torch.abs(igs_f)

# Default model
# Prediction class (many-to-one)
# Multivariate autoregressive model with LSTM layers
class MVAR_biLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_outputs):
        super(MVAR_biLSTM, self).__init__() # 상속한 nn.Module에서 RNN에 해당하는 init 실행
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size[0], num_layers, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(int(hidden_size[0]*2), hidden_size[1], num_layers, bidirectional=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(int(hidden_size[1]*2), hidden_size[2])
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size[2], num_outputs)

    def forward(self, x): 
        # input x : (BATCH, LENGTH, INPUT_SIZE) 입니다 (다양한 length를 다룰 수 있습니다.).
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        out, _ = self.lstm1(x) # output : (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE) tensors. (hn, cn)은 필요 없으므로 받지 않고 _로 처리합니다. 
        out    = F.tanh(out)
        out    = self.dropout1(out)
        out, _ = self.lstm2(out)
        out    = F.tanh(out)
        out    = self.dropout2(out)
        # 마지막 time step(sequence length)의 hidden state를 사용해 Class들의 logit을 반환합니다(hidden_size -> num_classes). 
        out    = self.fc1(out[:,-1,:])
        out    = F.relu(out)
        out    = self.dropout3(out)
        out    = self.fc2(out)
        
        return out

def train_model(model, trainset, testset=[], num_epochs=50, lr=0.001, n_batch=512, verbose=True, device=None):
    if device is None:
        device = ("cuda"
             if torch.cuda.is_available()
             else "mps"
             if torch.backends.mps.is_available()
             else "cpu")
    # 손실 함수와 옵티마이저 정의
    criterion = nn.MSELoss() # 회귀
    testloss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 학습
    if isinstance(trainset, torch.utils.data.DataLoader):
        dataset = trainset
        total_batch = len(dataset)
    else:
        x_train, y_train = trainset
        total_batch = np.ceil(x_train.shape[0]/n_batch).astype(int) # 배치 개수
        x_batch=[]
        y_batch=[]
        for b in range(total_batch):
            x_batch.append(x_train[b*n_batch:(b+1)*n_batch,:,:])
            y_batch.append(y_train[b*n_batch:(b+1)*n_batch,:])
    # CUDA timer
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_s = torch.cuda.Event(enable_timing=True)
    total_e = torch.cuda.Event(enable_timing=True)
    train_hist = np.zeros(0)
    test_hist = np.zeros(0)
    total_s.record()
    for epoch in range(num_epochs):
        start.record()
        avg_cost = 0.0
        model.train()
        if not isinstance(trainset, torch.utils.data.DataLoader):
            dataset=zip(x_batch, y_batch)
        for i, (x_part, y_part) in enumerate(dataset):
            # 순전파
            outputs = model(x_part.to(device))
            loss = criterion(outputs, y_part.to(device))

            # 역전파 & 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_cost += loss.item()/total_batch
            if (i+1)%10 == 0 and verbose:
                print(f"\r{i+1}/{total_batch}",end=' ')
                progress = int((i+1)/total_batch*100)
                for i in range(progress): print('.',end='')
        if verbose: print("")
        train_hist = np.append(train_hist, avg_cost)

        if len(testset)>0:
            #testout = model(x_test.to(device))
            testout, tloss = evaluate_testset(model, testset, testloss, 
                                              nbatch=n_batch, device=device)
            #tloss=testloss(testout, ytest.to(device))
            test_hist = np.append(test_hist, tloss)
            end.record()
            torch.cuda.synchronize()
            if verbose:
                print('Epoch [{}/{}]  Train_loss: {:.5f}  Test_loss: {:.5f}  Elapsed: {:.3f}s'.format(
                    epoch+1, num_epochs, avg_cost, tloss, start.elapsed_time(end)/1000))
            else:
                print('.', end='')
            if epoch>3:
                # test loss가 3회 연속 증가하면 학습 중지
                test_loss_increase=np.array([1 if dif>0 else 0 for dif in test_hist[-3:]-test_hist[-4:-1]])
                if tloss > test_hist[max([epoch-3,0]):epoch].mean() and test_loss_increase.sum()>2:
                    print('Early termination', end=' ')
                    break
        else:
            end.record()
            torch.cuda.synchronize()
            if verbose:
                print('Epoch [{}/{}]  Train_loss: {:.5f}  Elapsed: {:.3f}s'.format(
                    epoch+1, num_epochs, avg_cost, start.elapsed_time(end)/1000))
            else:
                print('.',end='')
    total_e.record()
    torch.cuda.synchronize()
    print('(Total elapsed time: {:.3f}s)'.format(total_s.elapsed_time(total_e)/1000))
    return train_hist, test_hist

def evaluate_testset(model, testset, loss_func=None, nbatch=512, device=None):
    if device is None:
        device=next(model.parameters()).device
    if isinstance(testset, torch.utils.data.DataLoader):
        dataset = testset
        tbatch = len(dataset)
        x_batch, y_batch = next(iter(dataset))
        y_pred=torch.zeros(0,y_batch.shape[-1])
    else:
        x_test, y_test = testset
        tbatch = np.ceil(x_test.shape[0]/nbatch).astype(int) # 배치 개수
        x_batch=[]
        y_batch=[]
        for b in range(tbatch):
            x_batch.append(x_test[b*nbatch:(b+1)*nbatch,:,:])
            y_batch.append(y_test[b*nbatch:(b+1)*nbatch,:])
        dataset=zip(x_batch, y_batch)
        y_pred=torch.zeros(0,y_test.shape[-1]).to('cpu')
    loss_total=0.0
    model.eval()
    for i, (x_part, y_part) in enumerate(dataset):
        pred_batch=model(x_part.to(device))
        pred_data=pred_batch.data.cpu()
        if loss_func is not None:
            loss_total+=loss_func(pred_data, y_part).item()/tbatch
        y_pred=torch.cat((y_pred,pred_data),dim=0)
    if loss_func is None:
        return y_pred
    else:
        return y_pred, loss_total

def evaluation_corr(model, X, nbatch=512, device=None):
    """
    X = tuple of inputs and outputs (x, y)
    """
    if device is None:
        device=next(model.parameters()).device
    model.eval()
    if isinstance(X, torch.utils.data.DataLoader):
        y=X.dataset.dataset.y[X.dataset.indices]
        y_pred=torch.zeros(0,y.shape[1])
        for x_batch,y_batch in X:
            y_pred=torch.cat((y_pred,model(x_batch.to(device)).data.cpu()),dim=0)
    else:
        x,y=X
        y_pred=torch.zeros(0,y.shape[1])
        for i in range(int(np.ceil(x.shape[0]/nbatch))):
            x_batch=x[i*nbatch:(i+1)*nbatch,:,:]
            y_pred=torch.cat((y_pred,model(x_batch.to(device)).data.cpu()),dim=0)
    #y=y.to(device)
    corr_total=torch.corrcoef(torch.stack((y_pred.reshape(-1),y.reshape(-1)),1).T)[0,1]
    corr=torch.zeros(y.shape[-1])
    for i in range(y.shape[-1]):
        corr[i]=torch.corrcoef(torch.stack((y_pred[:,i].cpu(),y[:,i].cpu()),1).T)[0,1]
    return corr_total, corr

def evaluation_linear(model, x, y, nbatch=512, device=None):
    y_pred=torch.zeros(0,y.shape[1])
    model.eval()
    for i in range(int(np.ceil(x.shape[0]/nbatch))):
        x_batch=x[i*nbatch:(i+1)*nbatch,:,:].transpose(1,2).flip(2).reshape(-1,x.shape[1]*x.shape[2])
        y_pred=torch.cat((y_pred,
            torch.matmul(model.reshape(-1,model.shape[1]*model.shape[2]),x_batch.T).T),dim=0)
    corr_total=torch.corrcoef(torch.stack((y_pred.reshape(-1),y.reshape(-1)),1).T)[0,1]
    corr=torch.zeros(y.shape[-1])
    for i in range(y.shape[-1]):
        corr[i]=torch.corrcoef(torch.stack((y_pred[:,i],y[:,i]),1).T)[0,1]
    return corr_total, corr

def display_conn_spectrum(conn, frq=None, title='', yscale=1.0, thresh=0):
    """
    Display connectivity spectrum for DTF, PDC, and IGC
    Parameters
    ----------
    conn : connectivity values (shape: frq, nch, nch)

    Returns : fig
    -------
    """
    nch=conn.shape[1]
    if frq is None:
        frq=np.arange(0,conn.shape[0])
    fig,axs=plt.subplots(nch,nch,dpi=300)
    fig.suptitle(title)
    fig.subplots_adjust(hspace=1.5)
    # x->y connectivity (a column causes a row)
    for x in range(nch):
        for y in range(nch):
            axs[y,x].bar(frq,conn[:,y,x],width=frq[1]-frq[0])
            if thresh>0:
                axs[y,x].plot(frq,np.ones(shape=frq.shape)*thresh, 'r')
            axs[y,x].set(xlim=(frq[0],frq[-1]),ylim=(0,yscale))
            axs[y,x].set_xticks(frq[0:-1:np.int32(frq.size/5)])
            if y<nch-1:
                axs[y,x].set_xticklabels([])
            else:
                axs[y,x].set_xticklabels(frq[0:-1:np.int32(frq.size/5)], rotation=45)
            axs[y,x].set_yticks(np.arange(0,yscale+1/3,yscale/2))
            if x>0:
                axs[y,x].set_yticklabels([])
            else:
                axs[y,x].set_yticklabels(labels=np.arange(0,yscale+1/3,yscale/2).round(1))
            axs[y,x].set_title('%d→%d' %(x+1,y+1))
    plt.show()

    return fig

def display_connectivity(conn, title='', vmin=0.0, vmax=1.0, labels=[]):
    """
    Display connectivity
    conn : connectivity values (shape: nch, nch)
    """
    if isinstance(conn,list) or isinstance(conn,tuple):
        nch=conn[0].shape[0]
        if len(labels)!=nch:
            labels = range(1,nch+1)
        if not isinstance(title,list) and not isinstance(title,tuple):
            title=[title for i in range(len(conn))]
        FONT_SCALE=conn[0].shape[-1]/10
        fig, axs = plt.subplots(1, len(conn), sharey=True)
        fig.set_size_inches((4+2*len(conn))*FONT_SCALE,10*FONT_SCALE)
        for i, (con, subtitle) in enumerate(zip(conn, title)):
            mapper=axs[i].pcolor(labels, labels, con, shading='nearest', 
                                 vmin=vmin, vmax=vmax, cmap='gist_heat_r')
            axs[i].axis('image')
            axs[i].set_xticks(np.arange(1,1+con.shape[0], np.ceil(con.shape[0]/10), dtype=int))
            axs[i].set_xticklabels(range(1, nch+1, int(np.ceil(nch/10))), rotation=45, fontsize=7*FONT_SCALE)
            axs[i].set_xlabel('From', fontsize=10*FONT_SCALE)
            axs[i].set_title(subtitle, fontsize=10*FONT_SCALE)
        axs[0].set_yticks(np.arange(1,1+con.shape[0], np.ceil(con.shape[0]/10), dtype=int))
        axs[0].set_yticklabels(range(1, nch+1, int(np.ceil(nch/10))), fontsize=7*FONT_SCALE)
        axs[0].set_ylabel('To', fontsize=10*FONT_SCALE)
        fig.colorbar(mapper, ax=axs, shrink=1/(len(conn)/2+2.5))
        #fig.show()
    else:
        nch = conn.shape[0]
        if len(labels)!=nch:
            labels = range(1,nch+1)
        plt.pcolor(labels, labels, conn, shading='nearest', vmin=vmin, vmax=vmax, cmap='gist_heat_r')
        plt.xticks(np.arange(1,1+con.shape[0], np.ceil(con.shape[0]/10), dtype=int), rotation=45)
        plt.xticklabels(range(1, nch+1, int(np.ceil(nch/10))), rotation=45, fontsize=7*FONT_SCALE)
        plt.yticks(np.arange(1,1+con.shape[0], np.ceil(con.shape[0]/10), dtype=int))
        plt.yticklabels(range(1, nch+1, int(np.ceil(nch/10))), fontsize=7*FONT_SCALE)
        plt.xlabel('From', fontsize=10*FONT_SCALE)
        plt.ylabel('To', fontsize=10*FONT_SCALE)
        plt.colorbar()
        plt.title(title, fontsize=10*FONT_SCALE)
    plt.show()

# rank correlation
# x=[N, P], Y=[N, Q]
# P and Q are number of features
# N is the number of samples
# output dimension is [P, Q]
def corr(x, y=None):
    if y is None:
        y = x
    # 벡터화된 방식으로 상관계수 계산
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean

    x_std = x_centered.std(dim=0, keepdim=True)
    y_std = y_centered.std(dim=0, keepdim=True)

    covariance = torch.matmul(x_centered.T, y_centered) / (x.shape[0] - 1)
    correlation = covariance / (x_std.T @ y_std)
    return correlation

# Parameter estimation using Yule Walker Equation
# x.shape should be Nsamples X Nfeatures
def yule_walker(x, order, flat=False):
    M=x.shape[1] # number of features
    N=x.shape[0] # number of samples
    r=torch.zeros(M,M,order+1)
    R=torch.zeros(M*order,M*order)
    rr=torch.zeros(M*order,M)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(0, order+1):
        r[:,:,i]=corr(x[:N-order,:], x[i:N-order+i,:])
    for a in range(order):
        for b in range(order):
            R[M*a:M*(a+1), M*b:M*(b+1)]=r[:,:,abs(a-b)]
        rr[M*a:M*(a+1),:]=r[:,:,a+1].T
    A=torch.matmul(torch.linalg.inv(R), rr)
    AA=torch.zeros(M,M,order)
    for i in range(order):
        AA[:,:,i]=A[M*i:M*(i+1),:]
    end.record()
    torch.cuda.synchronize()
    print('Computation time for linear regression: %.3fs' %(start.elapsed_time(end)/1000))
    return A if flat else AA

def yule_walker_VF(x, order, flat=False):
    M = x.shape[1]  # number of features
    N = x.shape[0]  # number of samples
    r = torch.zeros(M, M, order + 1)
    R = torch.zeros(M * order, M * order)
    rr = torch.zeros(M * order, M)

    start = time.time()

    # 전체 cross-correlation을 벡터화된 방식으로 계산
    x_subs = [x[i:N-order+i, :] for i in range(order+1)]
    x_stack = torch.stack(x_subs, dim=0)  # (order+1, N-order, M)

    # 벡터화된 corr 계산
    x0 = x_stack[0].T  # 기준 데이터 (M, N-order)
    x_means = x_stack.mean(dim=1, keepdim=True)  # (order+1, 1, M)
    x_stds = x_stack.std(dim=1, keepdim=True)  # (order+1, 1, M)
    x_centered = x_stack - x_means  # (order+1, N-order, M)

    # x0와 각 시점 데이터 간의 covariance 계산
    covariances = torch.einsum('ij,kjl->kil', x0, x_centered) / (N - order - 1)  # (order+1, M, M)
    x0_std = x_centered[0].std(dim=0, keepdim=True)  # (1, M)
    #correlation_matrices = covariances / (x0_std.T @ x_stds.squeeze(1))  # (order+1, M, M)
    correlation_matrices = covariances / torch.einsum('ij,kjl->kil', x0_std.T, x_stds)  # (order+1, M, M)

    r = correlation_matrices  # (order+1, M, M)

    # R와 rr를 벡터화하여 계산
    indices = torch.arange(order).repeat(order, 1)  # (order, order)
    abs_diff = torch.abs(indices - indices.T)  # |a-b| 계산 (order, order)
    # R 행렬 계산: r[|a-b|]로부터 블록 생성
    R_blocks = r[abs_diff]  # (order, order, M, M)
    R = R_blocks.permute(0, 2, 1, 3).reshape(M * order, M * order)
    # rr 행렬 계산
    rr_blocks = r[1:order+1]  # r[a+1] 가져오기 (order, M, M)
    rr = rr_blocks.permute(0, 2, 1).reshape(M * order, M)

    # A 행렬 계산
    A = torch.linalg.solve(R, rr)
    AA= A.permute(1,0).reshape([M,order,M]).permute(2,0,1)

    print('Computation time for linear regression: %.3fs' % (time.time() - start))
    return A if flat else AA

def DTF(A, sfreq=1000, foi=torch.arange(0,100,5), normalized=False):
    order=A.shape[2]
    M=A.shape[0]
    Af=torch.zeros(size=(M,M,len(foi)), dtype=torch.complex64)
    Aff=torch.zeros(size=(M,M,len(foi)), dtype=torch.complex64)
    for f_ind, f in enumerate(foi):
        for i in range(order):
            Af[:,:,f_ind]=Af[:,:,f_ind]+A[:,:,i]*torch.exp(-1j*2*torch.pi*f*i/sfreq)
        Aff[:,:,f_ind]=torch.eye(M)-Af[:,:,f_ind]
    dtf=torch.zeros(size=(M,M,len(foi)))
    for f in range(len(foi)):
        tmp=torch.abs(torch.linalg.inv(Aff[:,:,f]))
        dtf[:,:,f]=tmp/torch.sqrt(torch.diag(torch.matmul(torch.matmul(
            tmp,torch.eye(M)),tmp.T))).unsqueeze(1).repeat(1,tmp.shape[1])
    pdc=torch.zeros(size=(M,M,len(foi)))
    for f in range(len(foi)):
        tmp=torch.abs(Aff[:,:,f])
        pdc[:,:,f]=tmp/torch.sqrt(torch.diag(torch.matmul(torch.matmul(
            tmp.T,torch.eye(M)),tmp))).unsqueeze(1).repeat(1,Aff.shape[1]).T

    return dtf, pdc

def evaluate_connectivity_with_hitmap(con_mat, con_org, diag=False):
    M=con_mat.shape[0]
    con_thresh = con_mat.clone().detach()
    con_thresh = torch.where(con_thresh>0, 1.0, 0.0)
    if diag==False:
        con_thresh=torch.diagonal_scatter(con_thresh, torch.zeros(M), 0)
    recall=torch.sum(con_thresh*con_org)/torch.sum(con_org)
    precision=torch.sum(con_thresh*con_org)/torch.sum(con_thresh)
    accuracy=(torch.sum(torch.abs(con_thresh-1+con_org))-M) / (M*(M-1))
    F1=2*recall*precision/(recall+precision)
    performance={'recall': recall, 'precision': precision,
                 'accuracy': accuracy, 'F1': F1}
    
    return performance

class MVAR_biLSTM_simple(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_outputs):
        super(MVAR_biLSTM_simple, self).__init__() # 상속한 nn.Module에서 RNN에 해당하는 init 실행
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size[0], num_layers, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(int(hidden_size[0]*2), hidden_size[1])
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size[1], num_outputs)

    def forward(self, x): 
        # input x : (BATCH, LENGTH, INPUT_SIZE) 입니다 (다양한 length를 다룰 수 있습니다.).
        self.lstm1.flatten_parameters()
        out, _ = self.lstm1(x) # output : (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE) tensors. (hn, cn)은 필요 없음
        out    = F.tanh(out)
        out    = self.dropout1(out)
        # 마지막 time step(sequence length)의 hidden state를 사용
        out    = self.fc1(out[:,-1,:])
        out    = F.tanh(out)
        out    = self.dropout3(out)
        out    = self.fc2(out)
        
        return out

from typing import Tuple, Optional
class DM_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, layer_dim=1, bidirectional=False, bias=True, device=None):
        super(DM_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.layer_dim = layer_dim
        self.bias = bias
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size, bias, device=device)
        self.dm = nn.Parameter(torch.zeros(size=(seq_len, input_size, hidden_size)))
        self.dmb= nn.Parameter(torch.zeros(hidden_size))
        if bidirectional:
            self.lstm_rcell = nn.LSTMCell(input_size, hidden_size, bias, device=device)
        self.bidirectional = bidirectional
        self.reset_parameters()
        self.device = device
        
    def reset_parameters(self):
        self.dm.data.zero_()
        self.dmb.data.zero_()

    def forward(self, x, hidden:Optional[Tuple[torch.Tensor, torch.Tensor]]=None)->torch.Tensor:
        if self.device is None:
            self.device=next(self.lstm_cell.parameters()).device
        device=x.device
        
        if hidden is None:
            h0 = torch.zeros(x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(x.size(0), self.hidden_size).to(device)
        else:
            h0, c0 = hidden[0], hidden[1]
        
        if x.size(1)!=self.seq_len:
            print('Error: LSTM sequence length and input size are not matched')
            return torch.tensor([])
        
        outs  = torch.zeros([x.size(0), self.seq_len, self.hidden_size], device=device)
        outrs = torch.zeros(0)
        cn = c0[:,:]
        hn = h0[:,:]
        if self.bidirectional:
            outrs = torch.zeros([x.size(0), self.seq_len, self.hidden_size], device=device)
        cnr = c0[:,:]
        hnr = h0[:,:]

        # input: (batch, seq, input) ==> (seq, batch, input)
        # dm   : (seq, input, hidden)
        dhn = torch.matmul(torch.permute(x,(1,0,2)), self.dm)
        for seq in range(self.seq_len):
            hn, cn = self.lstm_cell(x[:, seq, :], (hn, cn))
            dout = dhn[seq] + self.dmb
            # dw linear layer는 batchxseq_len 으로 flatten 되어 있으므로,
            # seq_len 만큼 뛰어넘어가며 seq별 출력 데이터를 모아야 한다.
            hn = hn + F.tanh(dout)
            if self.bidirectional:
                seqr = self.seq_len-1-seq
                hnr, cnr = self.lstm_rcell(x[:, seqr, :], (hnr, cnr))
                dout = dhn[seqr] + self.dmb
                # reverse 방향으로 출력 데이터 취합
                hnr = hnr + F.tanh(dout)
                outrs[:,seq,:]=hnr
            outs[:,seq,:]=hn
        
        return torch.cat((outs, outrs), dim=-1)

class MVAR_DMLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, num_layers, num_outputs):
        super(MVAR_DMLSTM, self).__init__() # 상속한 nn.Module에서 RNN에 해당하는 init 실행
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = DM_LSTM(input_size, hidden_size[0], seq_len, num_layers, bidirectional=True)
        self.fc1 = nn.Linear(int(hidden_size[0]*2), hidden_size[1])
        self.fc2 = nn.Linear(hidden_size[1], num_outputs)

    def forward(self, x): 
        # input x : (BATCH, LENGTH, INPUT_SIZE) 입니다 (다양한 length를 다룰 수 있습니다.).
        out    = self.lstm1(x) # output : (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE) tensors. (hn, cn)은 필요 없음
        out    = F.tanh(out)
        # hidden state of the last time step(sequence length)
        out    = self.fc1(out[:,-1,:])
        out    = F.tanh(out)
        out    = self.fc2(out)
        
        return out
