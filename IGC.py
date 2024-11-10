import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_model_summary
import copy
from scipy import stats
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
        for i in range(order, sn):
          if nonlinear:
            x[i,nodes[1]]+=torch.sum([wei*x[i-a-1,nodes[0]]**(order-a) for a, wei in enumerate(weight)])
          else:
            x[i,nodes[1]]+=torch.sum(weight*x[i-order:i,nodes[0]].flip(0))
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
                      target_size, step, single_step=False):
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

def get_gradients(model, inputs):
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
        print('.', end='')
    print(f'\n{len(inputs)} gradients were collected')

    return grads

def get_integrated_gradients(model, x, baseline=None, num_steps=50, side=2):
    """
    igs = get_integrated_gradients(model, x, baseline=None, num_steps=100)
    model    : pytorch DNN model
    x        : reference input
    baseline : default (None) indicates 0
    num_steps: number of steps from baseline to the reference input
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
    for step in range(1,num_steps+1):
        interpolated_signal1=(baseline+(step/num_steps)*(x-baseline)).unsqueeze(0).clone().detach()
        interpolated_signal1.requires_grad_(True)
        inputs1.append(interpolated_signal1)
        if side==2:
            interpolated_signal2=(baseline+(step/num_steps)*(-x-baseline)).unsqueeze(0).clone().detach()
            interpolated_signal2.requires_grad_(True)
            inputs2.append(interpolated_signal2)
    
    # 2. Remove random effects from the model
    n_model = copy.deepcopy(model)
    for child in n_model.children():
        if isinstance(child, nn.Dropout):
            child.p = 0
    
    # 3. Get gradients
    grads1=get_gradients(n_model, inputs1)
    # Approximate the integral using the trapezoidal rule
    grads1 = (grads1[:-1] + grads1[1:]) / 2.0
    # 4. Calculate integrated gradients
    igs1 = (x - baseline).squeeze().unsqueeze(2).repeat(1,1,grads1.shape[-1]) * grads1.mean(0)

    if side==2:
        # For negative activity (Repeat 3~4)
        grads2=get_gradients(n_model, inputs2)
        grads2 = (grads2[:-1] + grads2[1:]) / 2.0
        igs2 = (-x - baseline).squeeze().unsqueeze(2).repeat(1,1,grads2.shape[-1]) * grads2.mean(0)
    
        igs=(igs1-igs2)/2
    else:
        igs = igs1

    return igs

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
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.bnorm1 = nn.BatchNorm1d(hidden_size*2)
        self.fc1 = nn.Linear(int(hidden_size*2), hidden_size)
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x): 
        # input x : (BATCH, LENGTH, INPUT_SIZE) 입니다 (다양한 length를 다룰 수 있습니다.).
        self.lstm1.flatten_parameters()
        out, _ = self.lstm1(x) # output : (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE) tensors. 
                               # We don't need (hn, cn)
        out    = F.tanh(out)
        out    = self.dropout1(out)
        out    = self.bnorm1(out.transpose(1,2)).transpose(1,2)
        # Extract the last time step(sequence length)
        out    = self.fc1(out[:,-1,:])
        out    = F.tanh(out)
        out    = self.dropout3(out)
        out    = self.fc2(out)
        
        return out

def train_model(model, trainset, testset=[], num_epochs=50, lr=0.001, n_batch=512, device=None):
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
        dataset=zip(x_batch, y_batch)
    # CUDA timer
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    train_hist = np.zeros(0)
    test_hist = np.zeros(0)
    for epoch in range(num_epochs):
        start.record()
        avg_cost = 0.0
        model.train()
        for i, (x_part, y_part) in enumerate(dataset):
            #x_part = x_part.to(device) # (BATCH(64), 10, 516)
            #y_part = y_part.to(device) # Size : (64 x 3)

            # 순전파
            outputs = model(x_part.to(device))
            loss = criterion(outputs, y_part.to(device))

            # 역전파 & 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_cost += loss.item()/total_batch
        train_hist = np.append(train_hist, avg_cost)
        end.record()
        torch.cuda.synchronize()

        if len(testset)>0:
            #testout = model(x_test.to(device))
            testout, tloss = evaluate_testset(model, testset, testloss, 
                                              nbatch=n_batch, device=device)
            #tloss=testloss(testout, ytest.to(device))
            test_hist = np.append(test_hist, tloss)
            print('Epoch [{}/{}]  Train_loss: {:.5f}  Test_loss: {:.5f}  Elapsed: {:.3f}s'.format(
                epoch+1, num_epochs, avg_cost, tloss, start.elapsed_time(end)/1000))
            if epoch>3:
                # test loss가 3회 연속 증가하면 학습 중지
                test_loss_increase=np.array([1 if dif>0 else 0 for dif in test_hist[-3:]-test_hist[-4:-1]])
                if tloss > test_hist[max([epoch-3,0]):epoch].mean() and test_loss_increase.sum()>2:
                    print('The model is being overfit. Iteration is terminated.')
                    break
        else:
            print('Epoch [{}/{}]  Train_loss: {:.5f}  Elapsed: {:.3f}s'.format(
                epoch+1, num_epochs, avg_cost, start.elapsed_time(end)/1000))
        #torch.cuda.empty_cache()
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
        y=X[1]
        x=X[0]
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

def ig_connectivity(x, model=None, order=20, foi=torch.arange(0,100,5), sfreq=1000, stat='np', nrs=0, side=2, device=None):
    """
    Integrated Gradients connectivity
    igc, igc_f = ig_connectivity(x, model=None, order=20, foi=torch.arange(0,100,10), sfreq=1000)
        x     : 2D input tensor (N samples x M channels)
        model : give a torch prediction model (if default None, the function uses a basic LSTM model)
        order : order of a regression model (default=20)
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
        igc   : representative connectivity between channels
        cl    : confidence level
        igc_t : T statistics of IGC
        igc_p : p-values of igc_t 
        igc_f : connectivity spectrum according to foi
        igc_o : original values of igc (time-series)
        igc_pdc: normalized IGC like PDC
        igc_dtf: noramlized IGC like DTF
        foi   : frequency of interest
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
        hist = train_model(model, (x_train, y_train), (x_test, y_test), num_epochs=50, device=device)
        # Use a trained model
        # Validation of model fitting
        #pred_y=model(x_train.to(device))
        pred_y = evaluate_testset(model, (x_train,y_train))
        corr_train=torch.corrcoef(torch.concatenate(
            (pred_y.reshape(1,-1),y_train.reshape(1,-1)),axis=0))
        corr_train=corr_train.cpu().detach().numpy()
        # Transform R to T
        corr_t = corr_train[0,1]*np.sqrt(torch.numel(pred_y)-2)/np.sqrt(1-corr_train[0,1]**2)
        # Estimate T to P
        corr_p = 2*(1 - stats.t.cdf(np.abs(corr_t), torch.numel(pred_y)-2))
        print("Train accuracy (CC) : %5.3f  (p=%7.5f)" %(corr_train[0,1], corr_p))

    # Connectivity measure
    ref = torch.ones((order, x.shape[-1]), dtype=torch.float)
    igc = get_integrated_gradients(model, ref, side=side)
    igc_pdc, igc_dtf, igc_f = igs_spectrum(igc, foi, sfreq, mode=2)
    
    igc_o = igc.transpose(1,2)
    igc = igc_f.mean(0)
    if stat=='np' or stat=='NP':
        pseudo_dist=torch.cat([igc[-nrs:,:].reshape(-1), igc[:-nrs,-nrs:].reshape(-1)], axis=0)
        igc = igc[:-nrs,:-nrs]
        igc_o = igc_o[:,:-nrs,:-nrs]
        igc_f = igc_f[:,:-nrs,:-nrs]
        cl=pseudo_dist.mean()-stats.t.ppf(
            0.01/igc.shape[0]**2/pseudo_dist.shape[0],pseudo_dist.shape[0]) * \
            pseudo_dist.std()/np.sqrt(pseudo_dist.shape[0])
        igc_t = (igc - pseudo_dist.mean())/torch.sqrt(pseudo_dist.var()/pseudo_dist.size(-1))
        igc_p = 2*(1 - stats.t.cdf(igc_t, pseudo_dist.size(-1)-1))*igc.shape[0]**2
        return igc, cl, igc_t, igc_p, igc_f, igc_o, foi
    
    return igc_pdc, igc_dtf, igc_o, igc, igc_f, foi

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
                axs[y,x].set_xticklabels(frq[0:-1:np.int32(frq.size/5)])
            axs[y,x].set_yticks(np.arange(0,yscale+1/3,1/2))
            if x>0:
                axs[y,x].set_yticklabels([])
            else:
                axs[y,x].set_yticklabels(labels=np.arange(0,yscale+1/3,1/2))
            axs[y,x].set_title('%d→%d' %(x+1,y+1))

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
        fig, axs = plt.subplots(1, len(conn), sharey=True)
        fig.set_size_inches(4+2*len(conn),10)
        for i, (con, subtitle) in enumerate(zip(conn, title)):
            mapper=axs[i].pcolor(labels, labels, con, shading='nearest', 
                                 vmin=vmin, vmax=vmax, cmap='Oranges')
            axs[i].axis('image')
            axs[i].set_xticks(range(0,0+con.shape[0]))
            axs[i].set_xticklabels(labels, rotation=45)
            axs[i].set_title(subtitle)
        fig.colorbar(mapper, ax=axs, shrink=1/(len(conn)/2+2.5))
        #fig.show()
    else:
        nch = conn.shape[0]
        if len(labels)!=nch:
            labels = range(1,nch+1)
        plt.pcolor(labels, labels, conn, shading='nearest', vmin=vmin, vmax=vmax, cmap='Oranges')
        plt.xticks(rotation=45)
        plt.colorbar()
        plt.title(title)
    plt.show()
