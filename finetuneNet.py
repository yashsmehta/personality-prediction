import jaximports
import importlib
importlib.reload(jaximports)
from jaximports import *

n_classes=2
batch_size = 32
do_darts = True
inp_dir, task, lr, epochs, seed, write_file, embed, layer = utils.parse_args()

start=time.time()
randkey = random.PRNGKey(seed)
rng = npr.RandomState(seed)

if (embed=='bert-base-uncased'):
    pretrained_weights='bert-base-uncased'
    n_hl=12
    hidden_dim=768

elif (embed=='bert-large-uncased'):
    pretrained_weights='bert-large-uncased'
    n_hl=24
    hidden_dim=1024

train_loc = 'pkl_data/glue-CoLA-train-'+pretrained_weights
test_loc = 'pkl_data/glue-CoLA-dev-'+pretrained_weights
# train_loc = 'pkl_data/imdb-train-'+pretrained_weights
# test_loc = 'pkl_data/imdb-test-'+pretrained_weights


f = open(train_loc, 'rb')
train_data = pickle.load(f)
train_x, train_y = list(zip(*train_data))
f.close()

f = open(test_loc, 'rb')
test_data = pickle.load(f)
test_x, test_y = list(zip(*test_data))
f.close()

def data_stream():
    while True:
      perm = rng.permutation(len(train_x))
      for i in range(len(train_x)):
        yield train_x[perm[i]], train_y[perm[i]]


batches = data_stream()

print('time to load dataset :', timedelta(seconds=int(time.time()-start)))

################################
#     dataloading done!
################################

def loss(params, alphaW, hidden_features, targets):
  inputs = get_final_feature_v(alphaW, hidden_features)
  preds = predict(params, inputs)
  return -jnp.mean(jnp.sum(preds * targets, axis=1))

@jit
def get_final_feature_v(alphaW, hidden_features):
    return jnp.einsum('k,kij->ij', alphaW, hidden_features)

@jit
def update(i, opt_state, alphaW, hidden_features, targets):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, alphaW, hidden_features, targets), opt_state)

@jit
def update_alphaW(alphaW, params, hidden_features, targets):    
    lr_alphaW=1e-2
    dw=grad(loss, argnums=1)(params, alphaW, hidden_features, targets)
    return alphaW - lr_alphaW * dw

def accuracy(params, hidden_features, targets):
  inputs=get_final_feature_v(alphaW, hidden_features)
  target_class = jnp.argmax(targets, axis=1)  
  predicted_class = jnp.argmax(predict(params, inputs), axis=1)
  return 100*jnp.mean(predicted_class == target_class)

def get_test_metrics(params, split):
    tmpa=0
    tmpl=0
    perm = rng.permutation(len(test_x))
    num=int(split*len(test_x))
    for j in range(num):
        hidden_features, targets = test_x[perm[j]], test_y[perm[j]]
        tmpa+=accuracy(params, hidden_features, targets)
        tmpl+=loss(params, alphaW, hidden_features, targets)

    return round(tmpa/num, 2), tmpl/num

#? define finetuning model!
init_random_params, predict = stax.serial(
    stax.Dense(500), Relu,
    stax.Dense(n_classes), LogSoftmax)

opt_init, opt_update, get_params=optimizers.adam(lr)

_, init_params = init_random_params(randkey, (-1, hidden_dim))
opt_state = opt_init(init_params)
itercount = itertools.count()

if(layer == 'all'):
    alphaW = np.full([n_hl], 1/n_hl)

else:
    alphaW = np.zeros([n_hl])
    alphaW[int(layer) - 1] = 1

start=time.time()
print('beginning training...')

acc = []
loss_val = []
step_alpha = 0
interval = 5
n_batches = len(train_x)
print(alphaW)

for epoch in range(epochs): 
    print('\nEPOCH ', epoch)
    epoch_start = time.time()
    for batch in range(n_batches):
        if(batch%50==0):    
            params = get_params(opt_state)
            tmpa, tmpl = get_test_metrics(params, split=1)
            acc.append(tmpa); loss_val.append(tmpl)
            print('acc : {}'.format(acc[-1]))
            print('loss : {}\n'.format(loss_val[-1]))
            
        hidden_features, targets = next(batches)
        
        if(batch%interval==0 and epoch < epochs-1 and do_darts):
            params = get_params(opt_state)
            alphaW = update_alphaW(alphaW, params, hidden_features, targets)

        opt_state = update(next(itercount), opt_state, alphaW, hidden_features, targets)

    print('epoch time :', timedelta(seconds=int(time.time()-epoch_start)))
    
print('training done!')
print('training time :', timedelta(seconds=int(time.time()-start)))


if (write_file):
    results_file='results.csv'
    meta_info=(lr, epochs, seed, embed, layer)
    utils.file_writer(results_file, meta_info, acc, loss_val)
# hidden_features || targets : (n_hl, 32, hidden_dim) || (32, 2)