# %%
import numpy as np
import random

import jax
import jax.numpy as jnp

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from flax import linen as nn

# %%
# Generate 100 sine functions with different a,b,c parameters and random noise
def sine(a,b,c,x):
    return a * np.sin(b * x + c)

x_data = np.arange(0,10,0.001)
a_list = np.linspace(0.1,10,100)
b_list = np.linspace(0.1,6,100)
c_list = np.linspace(0,5,100)

y_data = [0] * len(a_list)
signal = [0] * len(a_list)
dict = {}
for i in range(0,len(a_list)):
    random.shuffle(b_list)
    random.shuffle(c_list)
    if i <= 60:
        y_data[i] = sine(a_list[i], b_list[i], c_list[i], x_data) + np.random.normal(-a_list[i],a_list[i],len(x_data))
        signal[i] = 1   #boolean: 1 = there is a sine signal, 0 = just noise
    else:       
        y_data[i] = np.random.normal(-a_list[i],a_list[i],len(x_data))  # 40 noise datasets with no signal

y_data = np.array(y_data).transpose()
for n in range(0,len(x_data)):
    dict.update({str(x_data[n]): y_data[n]})

dict.update({'signal': signal})

df = pd.DataFrame(dict)
df = df.sample(frac=1)  # shuffle data

# %%
dfX = df.drop('signal',axis=1)
X = np.array(dfX)

n_classes = 2   # there is or isn't a sine signal

y = np.array(df['signal'])

# %%
from sklearn.preprocessing import StandardScaler

# %%
transformer = StandardScaler()
X = transformer.fit_transform(X)

# %%
m = X.shape[0]
test_frac = 0.2
test_sel = np.random.choice([True, False], size=m, p=[test_frac, 1-test_frac])
random.shuffle(test_sel)
X_tst = X[test_sel]
X_train = X[~test_sel]

y_tst = y[test_sel]
y_train = y[~test_sel]

# %%
X_train = jnp.array(X_train)
X_tst = jnp.array(X_tst)
y_train = jnp.array(y_train)
y_tst = jnp.array(y_tst)

# %%
batch_size = 20    
n_batches = X_train.shape[0] // batch_size

X_train = X_train[:n_batches * batch_size].reshape((n_batches, batch_size, *X_train.shape[1:]))
y_train = y_train[:n_batches * batch_size].reshape((n_batches, batch_size, *y_train.shape[1:]))

# %%
n_classes = 2

class DenseClassifier(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(10)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        x = nn.relu(x)
        x = nn.Dense(n_classes)(x)
        return x

# %%
dummy_input = jnp.ones((X_tst.shape[0], len(x_data)))
dnn = DenseClassifier()
print(dnn.tabulate(jax.random.PRNGKey(0), dummy_input))

# %%
from clu import metrics
from flax.training import train_state
from flax import struct
import optax

# %%
@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
   metrics: Metrics

# %%
def create_train_state(model, rng, learning_rate):
    params = model.init(rng, dummy_input)['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply, params=params, tx=tx,
        metrics=Metrics.empty())

# %%
# What does a step do
@jax.jit
def train_step(state, batch, label):
  def loss_fn(params):
    logits = state.apply_fn({'params': params}, batch)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=label).mean()
    return loss
  grad_fn = jax.grad(loss_fn)
  grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state

# %%
# For computing accuracy
@jax.jit
def compute_metrics(*, state, batch, label):
    logits = state.apply_fn({'params': state.params}, batch)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=label).mean()
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=label, loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state

# %%
init_rng = jax.random.PRNGKey(0)

learning_rate = 0.01

state = create_train_state(dnn, init_rng, learning_rate)
del init_rng  # Must not be used anymore.

# %%
metrics_history = {'train_loss': [],
                   'train_accuracy': [],
                   'test_loss': [],
                   'test_accuracy': []}

# %%
n_epochs = 50

# %%
step = 0

for _ in range(n_epochs):
  for batch, label in zip(X_train, y_train):

    # Run optimization steps over training batches and compute batch metrics
    state = train_step(state, batch, label) # get updated train state (which contains the updated parameters)
    state = compute_metrics(state=state, batch=batch, label=label) # aggregate batch metrics

    if (step+1) % n_batches == 0: # one training epoch has passed
      for metric,value in state.metrics.compute().items(): # compute metrics
        metrics_history[f'train_{metric}'].append(value) # record metrics
      state = state.replace(metrics=state.metrics.empty()) # reset train_metrics for next training epoch

      # Compute metrics on the test set after each training epoch
      test_state = state
      test_state = compute_metrics(state=test_state, batch=X_tst, label=y_tst)

      for metric,value in test_state.metrics.compute().items():
        metrics_history[f'test_{metric}'].append(value)

      print(f"train epoch: {(step+1) // n_batches}, "
            f"loss: {metrics_history['train_loss'][-1]}, "
            f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
      print(f"test epoch: {(step+1) // n_batches}, "
            f"loss: {metrics_history['test_loss'][-1]}, "
            f"accuracy: {metrics_history['test_accuracy'][-1] * 100}")
    step += 1

# %%
# Plot loss and accuracy in subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.set_title('Loss')
ax2.set_title('Accuracy')
for dataset in ('train','test'):
  ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
  ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
ax1.legend()
ax2.legend()
plt.show()
plt.clf()

# %% [markdown]
# # 2: Is your trained model confident that any of the stars in your test set are part of M4?

# %% [markdown]
# Yes, as seen above, my model is very confident that many of the stars are in M4, since there is a large spike at 4 on the plot above.

# %%
outputs = state.apply_fn({'params': state.params}, X_tst)
plt.scatter(df[test_sel]['signal'], jnp.diff(outputs, axis=1))

# %%
confident_pos = jnp.where(jnp.diff(outputs, axis=1) > 0)[0]
confident_df = df[test_sel].iloc[confident_pos]
confident_df

# %%
confident_neg = jnp.where(jnp.diff(outputs, axis=1) < 0)[0]
confident_ndf = df[test_sel].iloc[confident_neg]
confident_ndf

# %%
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

true_pos = confident_df.iloc[0].to_numpy()[:-1]
true_neg = confident_ndf.iloc[0].to_numpy()[:-1]
false_neg = confident_ndf.iloc[9].to_numpy()[:-1]

fig, ax = plt.subplots(figsize=(6, 4))

ax.scatter(x_data, true_pos, label='TP', zorder=3, s=1)
#plt.scatter(x_data, confident_df(84), label='FP')
ax.scatter(x_data, true_neg, label='TN', s=1)
ax.scatter(x_data, false_neg, label='FN', s=1)

sub_ax = inset_axes(parent_axes=ax, width="20%", height="17%")
sub_ax.scatter(x_data, true_pos, s=0.2)
sub_ax.set_xlim(0,5)

ax.set_xlabel('x')
ax.set_ylabel('y(x)')
ax.set_title('Neural Network Classification Examples')
ax.legend(loc = 'lower left');

# %%




