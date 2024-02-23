I want to build a model (ML, statistcs) for product recommendation in a mobile application.

# APPROACHES

## Collaborative Filtering:
This method makes automatic predictions (filtering) about the interests of a user by collecting preferences from many users (collaborating). The underlying assumption is that if a person A has the same opinion as a person B on an issue, A is more likely to have B's opinion on a different issue.

Types:
User-Based: Recommend products to a user that similar users have liked.
Item-Based: Recommend products similar to the ones the user has liked in the past.

## Content-Based Filtering:
This method uses item features to recommend additional items similar to what the user likes, based on their previous actions or explicit feedback.

### Neural Collaborative Filtering (NCF)
Implementacja NN

## Matrix Factorization Techniques (e.g., SVD, PCA, NMF):
These are particularly useful for sparse datasets and are a foundation for many recommendation systems. They reduce the dimensionality of the dataset by factorizing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices.

## Association Rule Mining (e.g., Market Basket Analysis):
This approach is used to uncover how items are associated with each other. It's based on the principle that if you buy a certain group of items, you are more (or less) likely to buy another group of items.

## Deep Learning Methods (e.g., Neural Collaborative Filtering, Autoencoders):
These methods can capture non-linear and complex structures and can be used to enhance the accuracy of recommendation systems.


# COMPARISON
The most effective approach often depends on the specific context and requirements of your application. A hybrid model or a more complex model like a neural network might offer the highest accuracy but at the cost of interpretability and computational resources. In contrast, simpler models like collaborative or content-based filtering might be more explainable and easier to implement but could lack in performance for certain tasks. It's often beneficial to start with simpler models to establish a baseline and progressively move to more complex models as needed.


# AI



Lets focus on Matrix Factorization Techniques and Neural Collaborative Filtering. Where do I run them? On Pytorch? Are there alternatives? I want to focus no neural networks.
