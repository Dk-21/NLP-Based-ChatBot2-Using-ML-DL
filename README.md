# Project Report Overview

## Introduction

This NLP project leverages traditional Machine Learning models alongside modern Natural Language Processing techniques to create a sophisticated chatbot. Our approach integrates models such as Naive Bayes (NB) and Support Vector Machines (SVM) with advanced structures, including encoders, decoders, and attention mechanisms. The goal is to provide an interactive and intelligent system capable of understanding and responding to user inquiries about Python programming.

## Objective

The primary objective of our project is to develop a robust Q&A system that can pre-emptively resolve coding queries, thus streamlining the development process and increasing productivity. This system is designed to assist novice and experienced programmers by providing immediate solutions to their coding issues and paving the way for future tools that enhance software development efficiency.

## Dataset

Glaive Code Assistant dataset: this dataset contains ~140k code problems and solutions designed to create intelligent Python code assistants. Structured in a Q&A format, this dataset contains real-world user questions worded for coding issues from the basics of data types to more complex object-oriented programming problems and features – approximately 60% being Python. By using this dataset, developers can create automated systems that can accurately respond to the queries posed by users in any given environment.

## Example of the Data

The dataset contains questions and answers related to Python programming, ranging from basic data types to complex object-oriented programming issues.

## System Description

### Encoder-Decoder Model (with Attention Mechanism)
**Purpose**: This model is designed to generate contextually appropriate responses based on user inputs, using sequence-to-sequence learning typically employed in machine translation and chatbot applications.

**Key Components**:
- **Encoder**: Processes the input text and converts it into a context vector.
- **Decoder**: Uses the context vector to generate output text step by step.
- **Attention Mechanism**: Enhances the model's ability to focus on relevant parts of the input during the decoding process, improving the relevance and specificity of responses.

**Integration in Chatbot**:
- **Role**: Acts as the primary response generation engine. When a user query is received, this model processes the text to generate a coherent and contextually relevant response.
- **Data Flow**: User inputs are pre-processed, tokenized, and fed into the encoder. The decoder then constructs a response, guided by the attention mechanism, which is returned to the user.

### Naive Bayes Classifier
**Purpose**: Utilized to classify user queries into predefined categories, which can help direct the query to the most suitable response mechanism within the chatbot.

**Key Components**:
- **TF-IDF Vectorization**: Converts text data into a format that can be efficiently processed by the Naive Bayes algorithm.
- **Naive Bayes Algorithm**: Predicts the category of the input query based on statistical inference.

**Integration in Chatbot**:
- **Role**: Acts as a preliminary filter to categorize user queries and decide if they can be answered directly with stored responses or if they need to be passed to the Encoder-Decoder model for generating a custom response.
- **Data Flow**: Incoming queries are vectorized and classified. Depending on the classification result, the query is either immediately answered or further processed by the Encoder-Decoder model.

### Support Vector Machine (SVM)
**Purpose**: Utilized for classifying user queries into predefined categories, which can help direct the query to the most suitable response mechanism within the chatbot.

**Key Components**:
- **TF-IDF Vectorization**: Converts text data into a format that can be efficiently processed by the SVM algorithm.
- **SVM Algorithm**: Predicts the category of the input query based on statistical inference.

**Integration in Chatbot**:
- **Role**: Acts as a preliminary filter to categorize user queries and decide if they can be answered directly with stored responses or if they need to be passed to the Encoder-Decoder model for generating a custom response.
- **Data Flow**: Incoming queries are vectorized and classified. Depending on the classification result, the query is either immediately answered or further processed by the Encoder-Decoder model.

## Specific NLP and ML Techniques Used

### Encoder-Decoder Model (with Attention Mechanism)
**Overview**: This segment implements an Encoder-Decoder architecture enhanced with an Attention mechanism, suitable for sequence-to-sequence learning, often used in machine translation and chatbot applications.

**Key Components**:
- **Encoder**: Uses LSTM (Long Short-Term Memory) units to process the input sequences and capture temporal dependencies. The encoder outputs a context vector representing the input sequence.
- **Decoder**: Also powered by LSTM units, it generates the output sequence step by step using the context vector and previous outputs.
- **Attention Mechanism**: Improves the model by allowing the decoder to focus on different parts of the encoder’s output for each step in the output sequence, thereby capturing nuances in longer sequences.

**Techniques**:
- **Text Pre-processing**: Includes converting characters from Unicode to ASCII, removing non-alphabetic characters, and handling contractions to clean and standardize the text.
- **Tokenization and Padding**: Converts text to sequences of integers and ensures that sequences are padded to a consistent length for modeling.
- **Embedding**: Transforms tokenized text into dense vectors that capture semantic meanings.
- **LSTM with Dropout**: Enhances the model's generalization by randomly dropping units (dropout) during training to prevent overfitting.

### Traditional ML Algorithms

#### Naive Bayes Classifier
**Overview**: Utilizes a Naive Bayes model for classifying text, a popular statistical technique for NLP tasks like spam detection and sentiment analysis, due to its simplicity and effectiveness.

**Techniques**:
- **TF-IDF Vectorization**: Transforms text into a meaningful vector of numbers based on the term frequency-inverse document frequency, which reflects the importance of words relative to the document and corpus.
- **Model Training and Prediction**: Employs the Naive Bayes algorithm, which assumes feature independence and calculates the probability of each category based on the Bayes theorem.

#### Support Vector Machine (SVM)
**Overview**: Implements an SVM for text classification. SVM is a powerful, supervised machine learning model that is effective in high-dimensional spaces and ideal for binary classification tasks.

**Techniques**:
- **Text Cleaning and Lemmatization**: This involves removing special characters and converting words to their base or dictionary form, reducing complexity and variability in the text.
- **SpaCy for Tokenization and Lemmatization**: Leverages the SpaCy library to process text for tokenization and lemmatization, helping to refine the text for further analysis.
- **TF-IDF Vectorization**: Like the Naive Bayes section, it prepares text data by converting it into TF-IDF vectors.
- **SVM Training**: Trains an SVM classifier with a linear kernel to distinguish between different classes based on the decision boundaries defined in the high-dimensional feature space.

## ChatBot

**Overview**: This part integrates user interaction and personalization into the chatbot, enhancing user experience by tailoring responses and maintaining a context of interaction.

**Techniques**:
- **Personal Data Handling**: Manages and utilizes user-specific data to personalize interactions, improving engagement.
- **Pattern Matching**: Uses regular expressions to identify user preferences (likes and dislikes) and other personal details from the conversation, which can influence the chatbot’s responses.
- **Named Entity Recognition (NER)**: Uses NER to extract information from user input.
- **Dynamic Response Generation**: The chatbot generates and outputs personalized responses based on the processed input and personal data.

## Data Cleaning for Different Methods

1. **Unicode Normalization**
   - **Function Used**: `unicode_to_ascii(s)`
   - **Purpose**: Converts Unicode characters to ASCII by removing diacritics from characters. This is important for standardizing text data and avoiding issues with character encoding.
   - **Method**: Utilizes Python’s ‘unicodedata.normalize’ function to decompose characters into their base characters and diacritics, then filters out diacritic marks using a list comprehension.

2. **Lowercasing and Stripping Whitespace**
   - **Applied In**: `preprocess_text(text)`
   - **Purpose**: Converts all text to lowercase to reduce complexity and variability, making the text uniform. Stripping removes any leading or trailing whitespace, cleaning up the input.
   - **Method**: Text strings are modified with the `.lower()` method for case normalization and `.strip()` method to remove extra spaces.

3. **Removing Non-Alphabetic Characters**
   - **Applied In**: `preprocess_text(text)`
   - **Purpose**: Cleans the text of punctuation, special characters, and other non-essential elements that might confuse the models.
   - **Method**: Uses regular expressions (`re.sub`) to replace non-word characters with spaces.

4. **Exclusion of Words Containing Numbers**
   - **Applied In**: `preprocess_text(text)`
   - **Purpose**: Removes words that contain digits, which are generally irrelevant for natural language understanding tasks.
   - **Method**: Another application of `re.sub` filters out any token that includes digits and ensures that only purely textual data is processed.

5. **Handling Contractions**
   - **Applied In**: `preprocess_text(text)`
   - **Purpose**: Expands contractions (e.g., changing “can’t” to “cannot”) to standardize text and improve the matching process during tokenization.
   - **Method**: A dictionary of contractions is used, and each word in the text is replaced with its expanded form if it exists in the dictionary.

6. **Removing Stopwords**
   - **In SVM and NB Sections**
   - **Purpose**: Stopwords (common words like 'the', 'is', 'at') are typically removed because they offer little value in understanding the intent of a question or response.
   - **Method**: Utilizes NLTK’s list of English stopwords and filters out these words from the processed text.

7. **Lemmatization**
   - **Applied In**: `lemmatize_text(text)` in SVM section
   - **Purpose**: Reduces words to their base or dictionary form (lemma) to treat different word forms as the same, reducing the complexity of the model’s input space.
   - **Method**: Utilizes SpaCy’s linguistic annotations to convert each token to its lemma.

8. **Text Vectorization (TF-IDF)**
   - **In SVM and NB Sections**
   - **Purpose**: Converts text into a numerical format that machine learning algorithms can process by weighing the terms based on their frequency and inverse document frequency across the corpus.
   - **Method**: `TfidfVectorizer` from scikit-learn transforms text into a sparse matrix of TF-IDF features.

9. **Named Entity Recognition (NER) Usage**
   - **Purpose**: Named Entity Recognition (NER) is used to identify and classify named entities mentioned in text into pre-defined categories such as names of people, organizations, locations, expressions of times, quantities, monetary values, percentages, etc. In the context of your chatbot, NER enhances user interaction by recognizing personal details, which can then be used to tailor the chatbot’s responses.
   - **Implementation Details**:
     - **Library Used**: SpaCy
     - **Model**: `en_core_web_sm`
     - **Function**: `extract_and_update_user_data(text, user_data)`

**Operational Steps**:
1. **Text Processing**:
   - The text input from the user is processed through SpaCy's NLP model. This model has been pre-trained on various text corpora and can perform several NLP tasks, including NER.

2. **Entity Extraction**:
   - As the text is processed, SpaCy identifies entities and classifies them into categories like PERSON (names), ORG (organizations), LOC (locations), and DATE (dates). Each identified entity is tagged with its corresponding category, which the chatbot can use to understand more about the user’s context or intent.

3. **Utilizing Extracted Entities**:
   - The extracted entities are used within the chatbot to personalize the conversation. For example, if a user mentions their name or a place, the chatbot can acknowledge this in subsequent responses, creating a more engaging and personalized interaction. Personal details extracted via NER are stored in a user-specific data structure (user_data). This information may include the user's name, associated locations, important dates, etc.

4. **Response Personalization**:
   - The chatbot uses the stored personal information to tailor its responses. For instance, if a user mentions their name, the chatbot might use this name in future interactions to create a more personalized and conversational experience.

## A Diagram of Logic


(Include a diagram here showing the logic flow of the chatbot, from user input to final response)
![Alt text]()


### Sample Dialog of Interaction

#### For Naive Bayes:

User: What is a list in Python?
Chatbot: A list in Python is a collection of items which is ordered and changeable. It allows duplicate members.


#### For SVM
User: Explain inheritance in Python.
Chatbot: Inheritance in Python is a feature that allows a class to inherit the attributes and methods of another class. This helps to reuse the code and create a hierarchical classification.


#### Encoder-Decoder Model (with Attention Mechanism)

User: How do I reverse a string in Python?
Chatbot: You can reverse a string in Python by using slicing: reversed_string = original_string[::-1].


## Appendix for User-Model
The JSON structure represents a user profile for a chatbot, detailing the user's preferences and personal information, which the chatbot can use to personalize interactions:

{
    "name": "Denish",
    "likes": [
        "coffee",
        "tea"
    ],
    "dislikes": [
        "tea"
    ],
    "personal_info": {
        "org": "University of Texas",
        "gpe": "Dallas"
    }
}

Here:
- "name": "Denish": This key-value pair holds the user's name, which the chatbot can use to address the user directly, making interactions feel more personal and engaging.
- "likes": ["coffee", "tea"]: This is an array of strings under the key "likes". It represents the things that the user, Denish, enjoys. The chatbot can use this information to make conversation about these beverages or suggest related topics.
- "dislikes": ["tea"]: Under the "dislikes" key is an array that lists things the user does not enjoy.
- "personal_info": { "org": "University of Texas", "gpe": "Dallas" }: The "personal_info" object contains structured data about the user's affiliations and location.


## Usage in Chatbot Interaction

- Customization: The chatbot can use these preferences to tailor dialogues, such as discussing programming topics related to Java or referencing events at the University of Texas.
- Recommendations: Based on his likes, the chatbot could recommend coffee brands, Java tutorials, or local events in Dallas.
- Engagement Strategies: To engage Denish effectively, the chatbot might initiate conversations by mentioning updates or news related to Java programming or inquiring about his experiences at the University of Texas.

## Implications for System Development

- Enhanced Personalization: This model exemplifies how detailed user profiles can facilitate deeper personalization, enhancing user experience by making interactions more relevant and engaging.

## Evaluation and Analysis

### Strengths:

- Personalization: The ability to learn user preferences and incorporate them into conversations using likes and dislikes from the conversation.
- Flexibility: Utilizes various NLP and advanced Machine Learning techniques for robust understanding and response generation using heavy models.
- Scalability: Designed to incorporate additional data sources and refine NLP capabilities easily.
- Speed: The speed of the response generation is quite good.

### Weaknesses:

- Context Understanding: Limited ability to maintain context over long conversations. The generated responses are sometimes not exactly related to the topics asked.
- Entity Recognition Limitations: May not always accurately capture or utilize personal information from the user.
- Dependence on Quality: The relevance of responses heavily relies on the training data. The more training data, the better it gets. The resources to train the model are not available, and the requirements are quite high, so I have to train on a very limited amount of data.

``` Performance and accuracy can be improved if a chance is given to train on more data. Integrating complex structures in addition to the current model can leverage ChatBot's performance to a very good level.```

### User Feedback (Survey Results of 10 people on an average basis):

- Ease of Interaction: 4.5 / 5
- Relevance of Responses: 4.5 / 5
- Personalization Effectiveness: 4 / 5
- User Feedback: “I find the Chatbot idea useful. We often search on Google for relevant information and spend an enormous amount of time reviewing several links. The chatbot eases the hunt by providing the most relevant information on the specified key input. Additionally, the GUI option of the Chatbot gives a modern feel to the end user and allows us to stay interested in using the application more times. One feature I would like to see is the ability to save our chat conversations for later access, thus allowing us to avoid going through the same conversations again with the Chatbot. Overall, I would rate this product 4/5, which excels in features and ease of use but could improve by introducing a conversation save feature.”

Feedback indicates a positive reception to the chatbot's personalization and ease of use, pointing out areas for improvement in understanding context and response relevance.

## Conclusion

This chatbot demonstrates the potential of NLP and Advanced Machine Learning technologies in creating interactive, personalized user experiences. Future work will focus on improving context management, training on more data, and refining entity recognition to enhance conversation quality and user satisfaction.

Feel free to replace the placeholders with your actual GitHub repository URL and adjust the details as necessary.




