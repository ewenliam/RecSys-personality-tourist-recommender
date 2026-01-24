Enhancing on Creating Profiles for a Tourist Recommendation System Based on  
User-generated POI Datasets Thesis 

Summary of the paper: 
The primary objective of the thesis is to develop a more personalized tourist 
recommendation system by integrating user and venue personalities (using the 
Myers-Briggs Type Indicator, MBTI) and topic modeling (via BERTopic) into the 
recommendation process. 

The thesis concludes that: 
● Personality-based and topic-driven features significantly improve recommendation 
accuracy compared to traditional methods (e.g., collaborative filtering). 
● BERT and BERTopic are effective for extracting personality traits and topics from 
unstructured text (user reviews and venue descriptions). 
● Challenges such as dataset imbalance (e.g., skewed MBTI distributions) were 
addressed via upsampling, but further data collection is needed for minority 
personality types. 
● Frequent closed itemset mining (DCI Closed) helped identify meaningful user-venue 
interaction patterns, though the system could benefit from hybrid models (e.g., 
combining with GNNs or multimodal NLP). 

Target Audience: 
The thesis is primarily academic , targeting researchers and students in the fields of: 
1. Recommendation systems (especially in tourism and location-based services). 
2. Natural Language Processing (NLP) , focusing on personality prediction and topic 
modeling. 
3. Data mining , particularly in the context of frequent itemset mining and hybrid models. 
the practical applications also align with businesses in the tourism industry: 
● Travel platforms 
● Destination marketing organizations

Expanding Thesis of Recommendation System. 
● GNN (graph Neural Network (GNNs)  
(Jiahao Zhang, Rui Xue, Wenqi Fan, Xin Xu, Qing Li, Jian Pei, and Xiaorui 
Liu. 2024. Linear-Time Graph Neural Networks for Scalable Recommendations. 
In Proceedings of the ACM Web Conference 2024 (WWW '24). Association for 
Computing Machinery, New York, NY, USA, 3533–3544. 
https://doi.org/10.1145/3589334.3645486) 
GNNs are a class of deep learning models designed to operate on graph-structured 
data. They generalize neural networks to non-Euclidean domains by propagating 
information through the edges of a graph to learn node embeddings. In 
recommendation systems, GNNs can model relationships between users, items 
(venues), and other entities (e.g., topics, locations, or time). 
● Reducing Overconfidence in Sequential Recommendation Trained with Negative 
Sampling (Based on the best paper on RecSys ACM conference 2023) 

WHY: 
Here we are using Graph Neural Networks (GNNs) to model user-venue interactions. GNNs 
often rely on sequential or temporal edges (e.g., "user X visited venue Y last week"). 
However, sequential models (including GNNs) trained with negative sampling can become 
overconfident: 
● Issue : GNNs might overemphasize strong patterns (e.g., "users who visit museums 
also visit historical sites") while ignoring less obvious but valid recommendations. 
● Solution from the Paper : 
    ○ Modified Loss Functions : The paper suggests using techniques like 
    confidence-aware loss or distribution calibration to penalize overconfident 
    predictions. 
    ○ Improved Negative Sampling : Better sampling strategies (e.g., harder 
    negatives) to ensure the model learns nuanced distinctions between relevant 
    and irrelevant items. 
● Example : GNN could avoid recommending only "popular museums" for an INTJ user 
by being less overconfident about their preferences and exploring niche venues. 

HOW: 
A. Modify Loss Functions 
● Current Approach : GNN/XGBoost models likely use standard loss functions (e.g., 
cross-entropy). 
● Enhancement : 
Implement confidence-aware loss (as in the paper) to penalize overconfident 
predictions. For example: 
Loss=CrossEntropy+λ⋅ConfidencePenalty(p) 
where p is the prediction confidence and λ balances the penalty. 
● Impact : Reduces overconfidence in GNN embeddings and XGBoost scores. 

B. Improve Negative Sampling 
● Current Approach : DCI Closed itemset mining and BERTopic might rely on basic 
negative sampling (e.g., random negatives). 
● Enhancement : 
Use hard negative mining (as in the paper) to select negatives that are semantically 
similar but irrelevant. For example: 
● A user who likes "museums" might get negative samples like "art galleries" 
(semantically related but not preferred). 
● Impact : Forces the model to learn subtle distinctions between similar venues. 

C. Uncertainty Estimation 
● Current Approach : BERT and GNN models output deterministic predictions. 
● Enhancement : 
○ Add uncertainty estimation layers (e.g., Bayesian neural networks or Monte 
Carlo dropout) to quantify confidence in recommendations. 
○ Use this uncertainty to diversify recommendations (e.g., include 
lower-confidence options to avoid over-reliance on top predictions). 
○ Example: A recommendation list might include both high-confidence (e.g., 
"Eiffel Tower") and mid-confidence (e.g., "hidden Parisian café") options.


Methodology Overview: Illuminating the Multimodal Path to Personalized Recommendations

1. The Big Picture: From Raw Data to Personalized Insights

The journey from raw digital noise to a "magical" user recommendation is a multi-stage pipeline designed to transform unstructured interactions into structured intelligence. Our foundation is the Yelp Dataset, comprising four critical sub-datasets: Business (metadata and attributes), Review (textual feedback), User (behavioral history), and Image (visual context).

To achieve state-of-the-art precision, our architecture converges these inputs into a Hybrid Recommendation Model through three specific mechanisms:

* Psychographic Feature Extraction: BERT-based classifiers process the Review dataset to extract MBTI personality traits, providing a "Who" for the model.
* Thematic Semantic Synthesis: Multimodal BERTopic blends Review text and Image embeddings to define the "What" of a venue.
* Relational Graph Mapping: Interaction history and metadata from the User and Business datasets are transformed into nodes and edges, forming a heterogeneous graph that encodes hidden patterns of preference.

Transitional Insight: Before a recommendation can be made, the system must first decode the "who" (the user’s psyche) and the "what" (the venue’s identity) through deep learning.


--------------------------------------------------------------------------------


2. Phase I: Decoding the User—BERT for MBTI Personality Prediction

To move beyond simple 1–5 star ratings, we utilize a BERT classifier (Bidirectional Encoder Representations from Transformers) to predict a user's Myers-Briggs Type Indicator (MBTI). This allows the system to understand the psychological drivers behind a review.

The BERT Classification Process

Component	Description
Input	User reviews from the Yelp Review dataset, truncated to a 512-token limit to meet Transformer engineering constraints.
Mechanism	A multi-layer Transformer architecture featuring bidirectional training, allowing the model to capture context-aware linguistic nuances from both directions simultaneously.
Output	Classification into one of 16 MBTI types (e.g., INTJ or ENFJ), creating a stable behavioral blueprint for the user.

The "So What?" Factor: Traditional rating systems are subjective and "noisy." By predicting personality, we gain a stable profile. For instance, an INFJ might prefer a quiet, library-style cafe. The system can suggest such venues even if the user has never visited one, simply because they align with the user's fundamental nature.

Transitional Insight: Once we understand the user’s personality, we must map it to the venues they visit by exploring the underlying themes of those locations.


--------------------------------------------------------------------------------


3. Phase II: Mapping the Venue—Multimodal BERTopic Modeling

To understand a venue’s "vibe," the system employs BERTopic. Unlike traditional Latent Dirichlet Allocation (LDA), which treats words as bags of tokens, BERTopic uses transformer-based embeddings to maintain semantic context.

The Multimodal Pipeline

1. Multimodal Input: The system synthesizes textual reviews with visual data. We utilize the CLIP model to generate image embeddings, allowing the architecture to "see" the venue’s atmosphere alongside the text.
2. Dimensionality Reduction: High-dimensional embeddings are processed via UMAP, compressing the data while preserving essential relational structures between venue features.
3. Clustering & Geospatial Mapping: The system applies K-Means to identify semantic groups (topics). Crucially, we use HDBSCAN separately for geospatial data, encoding coordinates as vectors to cluster venues into distinct regions (e.g., "Central Paris" or "The Strip").
4. Thematic Output: The pipeline generates context-aware topics such as "modern architecture" or "vegan-friendly vibes."

The "So What?" Factor: Text often captures functional details (the food), but images capture the aesthetic (the lighting, the seating). Multimodal synthesis provides a "richer understanding" that is critical for matching a venue’s aesthetic to a user’s personality.

Transitional Insight: These individual insights—personality and topics—are then woven together into a complex web of relationships.


--------------------------------------------------------------------------------


4. Phase III: The Architecture of Connection—Graph Construction and GNNs

The data is transformed into a heterogeneous Graph Structure to capture collaborative signals.

* Nodes: These include Users (tagged with MBTI), Venues (tagged with topics), and Context (Time, Weather, and Location).
* Edges: We define specific relationship types: User-Venue edges weighted by visit frequency/ratings; Venue-Topic edges for association strength; and Venue-Context edges to represent seasonal relevance (e.g., a "rooftop bar" node connected to a "Summer" context node).

The Linear-Time GNN (LTGNN)

To handle millions of interactions across the Yelp dataset, we implement the Linear-Time Graph Neural Network (LTGNN):

* Fixed-Point Iteration: Unlike traditional GNNs that stack multiple layers (causing "over-smoothing"), LTGNN uses a single propagation layer. It captures multi-hop info through information accumulation across training iterations using a fixed-point equation.
* EVR Sampling: We design an improved variance-reduced neighbor sampling strategy. This reduces the neighbor size during embedding aggregation, allowing the model to scale to massive graphs without sacrificing accuracy.
* Efficiency: This architecture ensures that computational complexity remains linear to the number of edges, making it viable for industrial-scale deployment.

Transitional Insight: The final step is turning this connected data into a prioritized, highly accurate recommendation list.


--------------------------------------------------------------------------------


5. Phase IV: The Hybrid Engine—Synthesis, Optimization, and Evaluation

The final engine combines the relational intelligence of the GNN with the classification power of XGBoost.

* DCI Closed Algorithm: This algorithm mines frequent closed itemsets from a user's history—sets where no superset has the same support. This creates compact user profiles that eliminate redundancy while preserving every meaningful behavioral pattern.
* XGBoost Refinement: We use XGBoost to synthesize GNN embeddings with specific context features (e.g., current weather) and personality traits to fine-tune the final suggestion.
* Overconfidence Mitigation (gBCE Loss): To prevent the model from suggesting only "popular" items, we replace standard loss with gBCE loss. We utilize a calibration parameter t (e.g., t=0.8) to balance the model’s confidence, ensuring that its predicted probability of a "match" aligns with actual user satisfaction.

Transitional Insight: This entire methodology transforms raw, noisy Yelp data into a curated, trustworthy user experience.


--------------------------------------------------------------------------------


6. Conclusion: The Learner’s Takeaway

As an AI architect, your success hinges on three pillars:

1. Multimodal Integration: Language (BERT) and vision (CLIP) must be treated as a single, unified signal to capture the true essence of a venue.
2. Algorithmic Efficiency: The Linear-Time GNN demonstrates that "deeper" is not always better. Mastering iteration-based information accumulation and variance-reduced sampling is key to scaling.
3. Human-Centric Modeling: Incorporating MBTI ensures your recommendations feel personal rather than purely transactional.

Modular AI architectures are the industry standard for a reason. By decoupling NLU, Graph, and Hybrid modules, development teams can achieve 25% faster progress through parallel development. This modularity allows you to upgrade your "Brain" (the NLU engine) or "Memory" (the GNN) independently, ensuring your system remains maintainable as it scales to millions of users.
s