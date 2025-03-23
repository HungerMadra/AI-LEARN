import torch
# Importing the PyTorch library for tensor operations
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

x_2 = inputs[1]     #1
d_in = inputs.shape[1]      #2
d_out = 2         #3

#1 The second input token serves as the query.
#2 The number of columns in the input tensor represents the dimensionality of the word embeddings.
#3 The output dimensionality of the query, key, and value vectors is set to 2.

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

#We set requires_grad=False to reduce clutter in the outputs, but if we were to use the weight matrices for model training, 
# we would set requires_grad=True to update these matrices during model training.

query_2 = x_2 @ W_query 
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value
# Calculate the query, key, and value vectors for the second word in the input sequence.
# The '@' operator is used for matrix multiplication in PyTorch.

print(query_2)
print(key_2)
print(value_2)

#The output tensors represent the query, key, and value vectors for the second word in the input sequence.
#The query vector is used to calculate the attention scores, while the key and value vectors are used to compute the weighted sum of the values.

keys = inputs @ W_key 
values = inputs @ W_value
print('keys.shape:', keys.shape)
print('values.shape:', values.shape)

# Calculate the key and value vectors for all words in the input sequence.
# The key and value vectors are used to compute the attention scores and the weighted sum of the values, respectively.

keys_2 = keys[1]             #1
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

#1 We extract the key vector for the second word in the input sequence to calculate the attention score between the query and the key.`
# The dot product is used to calculate the attention score.
# The attention score represents the similarity between the query and the key vectors.
# The higher the attention score, the more attention the model should pay to the corresponding value vector.
# The attention score is a scalar value that indicates the relevance of the key vector to the query vector.

attn_scores_2 = query_2 @ keys.T       #1
print(attn_scores_2)
#1 We calculate the attention scores for the second word in the input sequence by taking the dot product between the query vector and all key vectors.
# The dot product operation is performed using matrix multiplication.   
# The output tensor contains the attention scores for each word in the input sequence.
# The second word has the highest score, which is expected since the query is the second word.

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

# The softmax function is applied to the attention scores to obtain the attention weights.
# The softmax function normalizes the attention scores to obtain a probability distribution over the input sequence.
# The scaling factor d_k^0.5 is used to stabilize the training process and improve the interpretability of the attention weights.
