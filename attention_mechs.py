import torch

#create input tensor "your journey starts with one step"
#each word is represented by a 3-dimensional vector
#each row represents a word
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]   
#1 The second input token serves as the query.                         #1
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

#The output tensor contains the attention scores for each word in the input sequence. The second word has the highest score, which is expected since the query is the second word. 
# The dot product is used to calculate the attention scores. The dot product is a measure of similarity between two vectors. 
# The higher the dot product, the more similar the vectors are. In this case, the dot product is used to measure the similarity between the query and each word in the input sequence.

res = 0.
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]
print(res)
print(torch.dot(inputs[0], query))

#The output confirms that the sum of the element-wise multiplication gives the same results as the dot product
#The dot product is a more concise way to calculate the similarity between two vectors.


#We normalize each of the attention scores we computed previously. The main goal behind the normalization is to obtain attention weights that sum up to 1. 
# This normalization is a convention that is useful for interpretation and maintaining training stability in an LLM

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print('Attention weights:', attn_weights_2_tmp)
print('Sum:', attn_weights_2_tmp.sum())
