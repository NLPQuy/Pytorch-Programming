# Phase 2: NLP Specialization (Weeks 7-12)

## Learning Objectives
- Master text processing and embedding techniques
- Develop computational thinking for NLP tasks
- Build expertise in sequence modeling architectures
- Practice research-oriented NLP problem solving

## Week 7-8: Text Processing and Word Embeddings

### Learning Objectives
- Master text preprocessing and tokenization workflows
- Understand word embedding principles and implementations
- Build muscle memory for common NLP preprocessing patterns
- Develop foundation for advanced NLP architectures

### Muscle Memory Exercises

#### Exercise Set 1: Text Preprocessing Bootcamp
**Baseline Code Pattern:**
```python
import torch
import torch.nn as nn
import re
from collections import Counter
from typing import List, Dict, Tuple

class TextPreprocessor:
    def __init__(self, vocab_size: int = 10000, min_freq: int = 2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        
    def clean_text(self, text: str) -> str:
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def build_vocab(self, texts: List[str]) -> None:
        word_counts = Counter()
        for text in texts:
            words = self.clean_text(text).split()
            word_counts.update(words)
        
        idx = len(self.word2idx)
        for word, count in word_counts.most_common(self.vocab_size - 4):
            if count >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    def text_to_indices(self, text: str, max_length: int = None) -> List[int]:
        words = self.clean_text(text).split()
        indices = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in words]
        
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices.extend([self.word2idx["<PAD>"]] * (max_length - len(indices)))
        
        return indices
```

**Variations to Practice (Complete all 15):**

1. **Basic Tokenization Variants:**
   - Modify to handle different languages (add Unicode support)
   - Add support for subword tokenization (simple BPE)
   - Implement character-level tokenization
   - Add support for sentence-level tokenization
   - Implement word-piece tokenization basics

2. **Vocabulary Building Variants:**
   - Add frequency-based vocabulary pruning
   - Implement TF-IDF based vocabulary selection
   - Add support for multiple vocabulary sizes
   - Implement dynamic vocabulary expansion
   - Add vocabulary saving/loading functionality

3. **Advanced Preprocessing Variants:**
   - Add stemming and lemmatization
   - Implement n-gram tokenization (bigrams, trigrams)
   - Add support for named entity preservation
   - Implement custom stop word removal
   - Add text normalization for different domains (social media, formal text)

#### Exercise Set 2: Word Embedding Implementation Marathon
**Baseline Code Pattern:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Word2VecSkipGram(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Input and output embeddings
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings
        self.init_embeddings()
    
    def init_embeddings(self):
        initrange = 0.5 / self.embedding_dim
        self.in_embeddings.weight.data.uniform_(-initrange, initrange)
        self.out_embeddings.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, center_word: torch.Tensor, context_words: torch.Tensor):
        # Get embeddings
        center_embed = self.in_embeddings(center_word)  # [batch_size, embedding_dim]
        context_embed = self.out_embeddings(context_words)  # [batch_size, num_context, embedding_dim]
        
        # Compute scores
        scores = torch.bmm(context_embed, center_embed.unsqueeze(2)).squeeze(2)
        return scores

class SkipGramDataset(Dataset):
    def __init__(self, texts: List[List[int]], window_size: int = 2):
        self.pairs = []
        for text in texts:
            for i in range(len(text)):
                for j in range(max(0, i - window_size), min(len(text), i + window_size + 1)):
                    if i != j:
                        self.pairs.append((text[i], text[j]))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center), torch.tensor(context)
```

**Variations to Practice (Complete all 12):**

1. **Word2Vec Variants:**
   - Implement CBOW (Continuous Bag of Words) model
   - Add negative sampling to skip-gram
   - Implement hierarchical softmax
   - Add subsampling for frequent words

2. **GloVe Implementation:**
   - Build co-occurrence matrix construction
   - Implement GloVe objective function
   - Add bias terms and weighting function
   - Implement parallel GloVe training

3. **Modern Embedding Variants:**
   - Implement FastText with subword information
   - Add position-aware embeddings
   - Implement contextual embedding basics
   - Create domain-specific embedding training

#### Exercise Set 3: Attention Mechanism Foundations
**Baseline Code Pattern:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BasicAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, encoder_outputs: torch.Tensor, mask: torch.Tensor = None):
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = encoder_outputs.shape
        
        # Compute attention scores
        attention_scores = self.attention(encoder_outputs).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        # [batch_size, hidden_dim]
        
        return context, attention_weights

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: torch.Tensor = None):
        # Input shapes: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = query.shape
        
        # Compute attention scores
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d_model)
        # [batch_size, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.bmm(attention_weights, value)
        # [batch_size, seq_len, d_model]
        
        return output, attention_weights
```

**Variations to Practice (Complete all 10):**

1. **Attention Variants:**
   - Implement additive attention (Bahdanau)
   - Add multi-head attention mechanism
   - Implement self-attention for sequence modeling
   - Add relative position attention

2. **Advanced Attention:**
   - Implement sparse attention patterns
   - Add attention with learned position embeddings
   - Implement cross-attention for encoder-decoder
   - Create attention visualization utilities

### Thinking-Heavy Exercises

#### Exercise 1: Embedding Space Analysis
**Objective:** Understand the geometry and properties of word embedding spaces

**Tasks:**
1. **Semantic Similarity Analysis:**
   - Implement cosine similarity computation for word pairs
   - Create semantic analogy solver (king - man + woman = queen)
   - Build semantic clustering visualization
   - Analyze embedding space dimensionality effects

2. **Embedding Quality Evaluation:**
   - Implement intrinsic evaluation metrics
   - Create word similarity benchmarks
   - Build bias detection in embeddings
   - Design embedding space probing tasks

3. **Cross-lingual Embedding Alignment:**
   - Implement embedding space alignment techniques
   - Create bilingual dictionary induction
   - Build cross-lingual similarity metrics
   - Design multilingual embedding evaluation

#### Exercise 2: Tokenization Strategy Design
**Objective:** Design optimal tokenization strategies for different NLP tasks

**Tasks:**
1. **Domain-Specific Tokenization:**
   - Design tokenizer for social media text
   - Create scientific paper tokenization strategy
   - Build code tokenization for programming languages
   - Design tokenizer for multilingual settings

2. **Subword Tokenization Implementation:**
   - Implement Byte-Pair Encoding (BPE) from scratch
   - Create SentencePiece-style tokenization
   - Build dynamic vocabulary expansion
   - Design context-aware tokenization

3. **Tokenization Evaluation Framework:**
   - Create tokenization quality metrics
   - Build compression ratio analysis
   - Design downstream task evaluation
   - Implement tokenization robustness testing

#### Exercise 3: Attention Pattern Analysis
**Objective:** Understand and visualize attention mechanisms in depth

**Tasks:**
1. **Attention Visualization:**
   - Create attention heatmap generators
   - Build attention head analysis tools
   - Design attention pattern clustering
   - Implement attention flow visualization

2. **Attention Mechanism Comparison:**
   - Compare different attention variants on same task
   - Analyze computational complexity trade-offs
   - Design attention efficiency benchmarks
   - Create attention interpretability metrics

3. **Custom Attention Design:**
   - Design task-specific attention mechanisms
   - Create structured attention patterns
   - Build attention with external knowledge
   - Implement attention regularization techniques

### Projects

#### Project 2.1: Advanced Text Classification System
**Objective:** Build a comprehensive text classification system with modern NLP techniques

**Requirements:**
- **Data:** Multi-domain text classification (news, reviews, scientific abstracts)
- **Architecture:** Hierarchical attention network with pre-trained embeddings
- **Features:** Multi-task learning, domain adaptation, uncertainty quantification
- **Evaluation:** Cross-domain generalization, few-shot learning capabilities

**Key Components:**
1. **Text Preprocessing Pipeline:**
   - Multi-language support with language detection
   - Domain-specific tokenization strategies
   - Hierarchical text representation (word → sentence → document)
   - Data augmentation for low-resource domains

2. **Embedding Layer:**
   - Pre-trained word embeddings (Word2Vec, GloVe, FastText)
   - Fine-tuning strategies for domain adaptation
   - Embedding dropout and regularization
   - Dynamic vocabulary handling for new domains

3. **Attention Mechanisms:**
   - Word-level attention within sentences
   - Sentence-level attention within documents
   - Cross-domain attention transfer
   - Attention weight analysis and visualization

4. **Advanced Features:**
   - Multi-task learning across domains
   - Meta-learning for few-shot classification
   - Confidence estimation and calibration
   - Model interpretability and explainability

#### Project 2.2: Neural Language Model with Custom Architecture
**Objective:** Implement a language model that demonstrates understanding of linguistic structure

**Requirements:**
- **Data:** Multi-genre text corpus (fiction, news, academic, social media)
- **Architecture:** Custom neural architecture combining different modeling approaches
- **Features:** Syntax-aware modeling, controllable generation, evaluation metrics
- **Innovation:** Novel architectural components or training techniques

**Key Components:**
1. **Architecture Design:**
   - Combine RNN and attention mechanisms
   - Hierarchical modeling of text structure
   - Syntax-aware representations
   - Memory-augmented architecture for long-range dependencies

2. **Training Innovations:**
   - Custom loss functions for linguistic properties
   - Curriculum learning from simple to complex texts
   - Multi-task training with auxiliary objectives
   - Advanced optimization techniques

3. **Generation and Control:**
   - Controllable text generation with attributes
   - Style transfer between text genres
   - Coherence-aware generation strategies
   - Human evaluation framework

4. **Evaluation Framework:**
   - Perplexity analysis across different genres
   - Linguistic quality assessment (grammar, coherence)
   - Generation diversity and creativity metrics
   - Human preference evaluation

#### Project 2.3: Information Extraction with Neural Networks
**Objective:** Build an information extraction system using neural sequence labeling

**Requirements:**
- **Data:** Named Entity Recognition, Relation Extraction, Event Extraction
- **Architecture:** BiLSTM-CRF with attention and character-level features
- **Features:** Multi-task learning, domain transfer, active learning
- **Deployment:** Real-time inference with efficiency optimizations

**Key Components:**
1. **Sequence Labeling Architecture:**
   - Character-level and word-level representations
   - Bidirectional LSTM with highway connections
   - CRF layer for structured prediction
   - Multi-task learning across different IE tasks

2. **Feature Engineering:**
   - Linguistic features (POS tags, dependency relations)
   - Contextual features from large windows
   - External knowledge integration (gazetteers, KB)
   - Cross-sentence context modeling

3. **Domain Adaptation:**
   - Transfer learning between domains
   - Active learning for annotation efficiency
   - Domain adversarial training
   - Meta-learning for few-shot IE

4. **System Integration:**
   - Pipeline vs joint modeling approaches
   - Error propagation analysis and mitigation
   - Confidence estimation for predictions
   - User interface for annotation and correction

## Week 9-10: Recurrent Neural Networks and Sequence Modeling

### Learning Objectives
- Master RNN architectures and their variants
- Understand sequence-to-sequence modeling
- Build expertise in handling variable-length sequences
- Develop skills in temporal pattern recognition

### Muscle Memory Exercises

#### Exercise Set 1: RNN Architecture Mastery
**Baseline Code Pattern:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class VanillaRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, 
                 nonlinearity: str = 'tanh', dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                         nonlinearity=nonlinearity, dropout=dropout, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, input_size)
        
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        # x: [batch_size, seq_len, input_size]
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # RNN forward pass
        rnn_out, hidden = self.rnn(x, hidden)
        # rnn_out: [batch_size, seq_len, hidden_size]
        
        # Project to output size
        output = self.output_proj(rnn_out)
        # output: [batch_size, seq_len, input_size]
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gate parameters
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)  # Forget gate
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)  # Input gate
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)  # Output gate
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)  # Cell candidate
        
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        # x: [batch_size, input_size]
        # h_prev, c_prev: [batch_size, hidden_size]
        
        combined = torch.cat([x, h_prev], dim=1)
        
        # Compute gates
        f_t = torch.sigmoid(self.W_f(combined))  # Forget gate
        i_t = torch.sigmoid(self.W_i(combined))  # Input gate
        o_t = torch.sigmoid(self.W_o(combined))  # Output gate
        c_tilde = torch.tanh(self.W_c(combined))  # Cell candidate
        
        # Update cell state
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Update hidden state
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t
```

**Variations to Practice (Complete all 15):**

1. **Basic RNN Variants:**
   - Implement GRU cell from scratch
   - Add bidirectional RNN processing
   - Implement deep RNN with residual connections
   - Add layer normalization to RNN cells
   - Create custom activation functions for RNNs

2. **LSTM Enhancements:**
   - Implement peephole connections in LSTM
   - Add dropout within LSTM cells
   - Implement coupled input and forget gates
   - Create LSTM with attention mechanisms
   - Add highway connections for deep LSTMs

3. **Advanced RNN Architectures:**
   - Implement Clockwork RNN
   - Create hierarchical RNN structures
   - Add skip connections in RNN layers
   - Implement RNN with external memory
   - Create multi-timescale RNN architectures

#### Exercise Set 2: Sequence-to-Sequence Mastery
**Baseline Code Pattern:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple, Optional

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_seq: torch.Tensor, input_lengths: torch.Tensor):
        # input_seq: [batch_size, max_seq_len]
        # input_lengths: [batch_size]
        
        embedded = self.dropout(self.embedding(input_seq))
        # embedded: [batch_size, max_seq_len, embedding_dim]
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu(), 
                                                   batch_first=True, enforce_sorted=False)
        
        # LSTM forward
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        return output, (hidden, cell)

class AttentionDecoderRNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 encoder_hidden_size: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = nn.Linear(hidden_size + encoder_hidden_size, encoder_hidden_size)
        self.attention_combine = nn.Linear(encoder_hidden_size + embedding_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_token: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor], 
                encoder_outputs: torch.Tensor, mask: torch.Tensor = None):
        # input_token: [batch_size, 1]
        # hidden: ([num_layers, batch_size, hidden_size], [num_layers, batch_size, hidden_size])
        # encoder_outputs: [batch_size, encoder_seq_len, encoder_hidden_size]
        
        batch_size = input_token.size(0)
        
        embedded = self.dropout(self.embedding(input_token))
        # embedded: [batch_size, 1, embedding_dim]
        
        # Attention mechanism
        hidden_state = hidden[0][-1].unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Compute attention scores
        attention_scores = self.compute_attention(hidden_state, encoder_outputs, mask)
        # attention_scores: [batch_size, 1, encoder_seq_len]
        
        # Apply attention to encoder outputs
        context = torch.bmm(attention_scores, encoder_outputs)
        # context: [batch_size, 1, encoder_hidden_size]
        
        # Combine input and context
        combined_input = torch.cat([embedded, context], dim=2)
        combined_input = self.attention_combine(combined_input)
        
        # LSTM forward
        output, hidden = self.lstm(combined_input, hidden)
        
        # Final output projection
        output = self.out(output)
        # output: [batch_size, 1, vocab_size]
        
        return output, hidden, attention_scores
    
    def compute_attention(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor, 
                         mask: torch.Tensor = None):
        batch_size, encoder_seq_len, encoder_hidden_size = encoder_outputs.shape
        
        # Expand hidden state
        hidden_expanded = hidden.expand(-1, encoder_seq_len, -1)
        
        # Compute attention energy
        energy = torch.tanh(self.attention(
            torch.cat([hidden_expanded, encoder_outputs], dim=2)
        ))
        
        # Compute attention scores
        attention_scores = torch.sum(energy, dim=2, keepdim=True)
        attention_scores = attention_scores.transpose(1, 2)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e10)
        
        # Softmax
        attention_weights = F.softmax(attention_scores, dim=2)
        
        return attention_weights
```

**Variations to Practice (Complete all 12):**

1. **Encoder Variants:**
   - Implement bidirectional encoder
   - Add hierarchical encoding (word → sentence → document)
   - Create multi-scale encoder with different granularities
   - Implement encoder with self-attention layers

2. **Decoder Variants:**
   - Implement beam search decoding
   - Add coverage mechanism to attention
   - Create pointer-generator decoder
   - Implement decoder with copy mechanism

3. **Training Techniques:**
   - Implement teacher forcing vs scheduled sampling
   - Add length normalization for beam search
   - Create reinforcement learning for sequence generation
   - Implement minimum risk training

#### Exercise Set 3: Advanced Sequence Processing
**Baseline Code Pattern:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, 
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           bidirectional=True, dropout=dropout, batch_first=True)
        
        # Project back to original hidden size
        self.projection = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None):
        # x: [batch_size, seq_len, input_size]
        batch_size, seq_len, _ = x.shape
        
        if lengths is not None:
            # Pack padded sequence
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), 
                                                       batch_first=True, enforce_sorted=False)
            output, (hidden, cell) = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, (hidden, cell) = self.lstm(x)
        
        # output: [batch_size, seq_len, hidden_size * 2]
        
        # Project to original hidden size
        output = self.projection(output)
        # output: [batch_size, seq_len, hidden_size]
        
        return output, (hidden, cell)

class HierarchicalAttention(nn.Module):
    def __init__(self, hidden_size: int, attention_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        # Word-level attention
        self.word_attention = nn.Linear(hidden_size, attention_size)
        self.word_context = nn.Linear(attention_size, 1, bias=False)
        
        # Sentence-level attention
        self.sentence_attention = nn.Linear(hidden_size, attention_size)
        self.sentence_context = nn.Linear(attention_size, 1, bias=False)
        
    def forward(self, word_outputs: torch.Tensor, sentence_lengths: torch.Tensor):
        # word_outputs: [batch_size, max_sentences, max_words, hidden_size]
        # sentence_lengths: [batch_size, max_sentences]
        
        batch_size, max_sentences, max_words, hidden_size = word_outputs.shape
        
        # Reshape for word-level attention
        word_outputs_flat = word_outputs.view(-1, max_words, hidden_size)
        sentence_lengths_flat = sentence_lengths.view(-1)
        
        # Word-level attention
        sentence_vectors = []
        for i in range(batch_size * max_sentences):
            if sentence_lengths_flat[i] > 0:
                words = word_outputs_flat[i, :sentence_lengths_flat[i]]  # [actual_words, hidden_size]
                
                # Compute word attention
                word_attn = torch.tanh(self.word_attention(words))
                word_scores = self.word_context(word_attn).squeeze(-1)
                word_weights = F.softmax(word_scores, dim=0)
                
                # Weighted sum
                sentence_vec = torch.sum(word_weights.unsqueeze(-1) * words, dim=0)
                sentence_vectors.append(sentence_vec)
            else:
                sentence_vectors.append(torch.zeros(hidden_size, device=word_outputs.device))
        
        # Stack sentence vectors
        sentence_matrix = torch.stack(sentence_vectors).view(batch_size, max_sentences, hidden_size)
        
        # Sentence-level attention
        doc_vectors = []
        for i in range(batch_size):
            # Get actual sentences for this document
            actual_sentences = max_sentences  # Simplified - in practice, track document lengths
            sentences = sentence_matrix[i, :actual_sentences]
            
            # Compute sentence attention
            sent_attn = torch.tanh(self.sentence_attention(sentences))
            sent_scores = self.sentence_context(sent_attn).squeeze(-1)
            sent_weights = F.softmax(sent_scores, dim=0)
            
            # Weighted sum
            doc_vec = torch.sum(sent_weights.unsqueeze(-1) * sentences, dim=0)
            doc_vectors.append(doc_vec)
        
        document_representations = torch.stack(doc_vectors)
        
        return document_representations
```

**Variations to Practice (Complete all 10):**

1. **Bidirectional Processing:**
   - Implement different fusion strategies for bidirectional outputs
   - Add residual connections in bidirectional networks
   - Create attention-based fusion of forward/backward states
   - Implement multi-layer bidirectional processing

2. **Hierarchical Modeling:**
   - Create word → sentence → paragraph → document hierarchy
   - Implement different aggregation methods at each level
   - Add cross-level attention mechanisms
   - Design dynamic hierarchy based on content structure

### Thinking-Heavy Exercises

#### Exercise 1: RNN Memory Analysis
**Objective:** Understand and analyze memory mechanisms in recurrent networks

**Tasks:**
1. **Gradient Flow Analysis:**
   - Implement gradient norm tracking through time
   - Analyze vanishing/exploding gradient problems
   - Compare gradient flow in LSTM vs GRU vs vanilla RNN
   - Design gradient clipping strategies

2. **Memory Capacity Studies:**
   - Test long-term memory retention in different RNN variants
   - Analyze information bottlenecks in sequence processing
   - Design tasks to probe specific memory capabilities
   - Create memory-augmented RNN architectures

3. **Hidden State Interpretability:**
   - Visualize hidden state evolution over time
   - Probe hidden states for linguistic information
   - Analyze attention patterns in sequence models
   - Design interpretability metrics for RNN representations

#### Exercise 2: Sequence Generation Strategy Design
**Objective:** Design and analyze different sequence generation strategies

**Tasks:**
1. **Decoding Algorithm Comparison:**
   - Implement and compare greedy, beam search, nucleus sampling
   - Analyze trade-offs between quality and diversity
   - Design adaptive decoding strategies
   - Create evaluation metrics for generation quality

2. **Controllable Generation:**
   - Design conditioning mechanisms for style control
   - Implement content planning for structured generation
   - Create attribute-based generation systems
   - Design interactive generation interfaces

3. **Evaluation Framework Design:**
   - Create comprehensive generation evaluation metrics
   - Design human evaluation protocols
   - Implement automatic quality assessment
   - Build generation robustness testing

### Projects

#### Project 2.4: Neural Machine Translation System
**Objective:** Build a complete neural machine translation system with attention mechanisms

**Requirements:**
- **Data:** Multi-language pairs (English ↔ French, German, Spanish)
- **Architecture:** Transformer-style encoder-decoder with custom attention
- **Features:** Subword tokenization, beam search, BLEU evaluation
- **Advanced:** Back-translation, multilingual modeling, domain adaptation

**Key Components:**
1. **Data Processing Pipeline:**
   - Multilingual tokenization with SentencePiece
   - Data cleaning and filtering strategies
   - Parallel data alignment and quality assessment
   - Domain-specific data preparation

2. **Model Architecture:**
   - Custom encoder-decoder with attention
   - Positional encoding for sequence information
   - Multi-head attention mechanisms
   - Advanced optimization techniques

3. **Training Strategies:**
   - Curriculum learning from simple to complex sentences
   - Back-translation for monolingual data utilization
   - Multi-task learning across language pairs
   - Domain adaptation techniques

4. **Evaluation and Analysis:**
   - Automatic metrics (BLEU, METEOR, BERTScore)
   - Human evaluation protocols
   - Error analysis and linguistic evaluation
   - Attention visualization and interpretation

#### Project 2.5: Dialogue System with Memory
**Objective:** Create a dialogue system that maintains conversational context and memory

**Requirements:**
- **Data:** Multi-turn conversation datasets (PersonaChat, Wizard of Oz)
- **Architecture:** Memory-augmented sequence-to-sequence model
- **Features:** Persona consistency, emotion tracking, response diversity
- **Evaluation:** Multi-turn coherence, engagement metrics

**Key Components:**
1. **Memory Mechanisms:**
   - External memory bank for long-term information
   - Attention-based memory retrieval
   - Memory updating strategies
   - Persona and context integration

2. **Dialogue Management:**
   - Intent recognition and slot filling
   - Response generation with controllable attributes
   - Context tracking across multiple turns
   - Emotion and sentiment awareness

3. **Training Framework:**
   - Multi-task learning (response generation + auxiliary tasks)
   - Reinforcement learning for dialogue optimization
   - Adversarial training for response quality
   - Human-in-the-loop training

4. **Evaluation System:**
   - Automatic evaluation metrics (perplexity, diversity)
   - Human evaluation protocols (coherence, engagement)
   - Long-term conversation analysis
   - Persona consistency evaluation

#### Project 2.6: Text Summarization with Hierarchical Attention
**Objective:** Build an extractive and abstractive summarization system using hierarchical models

**Requirements:**
- **Data:** News articles, scientific papers, multi-document collections
- **Architecture:** Hierarchical encoder with extractive and abstractive components
- **Features:** Content selection, abstraction strategies, factual consistency
- **Evaluation:** ROUGE scores, factual accuracy, human assessment

**Key Components:**
1. **Hierarchical Encoding:**
   - Word-level encoding with BiLSTM
   - Sentence-level encoding and selection
   - Document-level representation learning
   - Cross-document information integration

2. **Extractive Component:**
   - Sentence scoring and ranking
   - Content selection with coverage constraints
   - Redundancy detection and removal
   - Position and importance modeling

3. **Abstractive Component:**
   - Pointer-generator network for handling OOV
   - Copy mechanism for factual accuracy
   - Content planning for coherent generation
   - Length and style control mechanisms

4. **Quality Assurance:**
   - Factual consistency checking
   - Coherence and fluency evaluation
   - Information coverage analysis
   - Bias detection and mitigation

## Week 11-12: Advanced NLP Training Techniques

### Learning Objectives
- Master advanced training strategies for NLP models
- Understand transfer learning and domain adaptation
- Build expertise in model optimization and regularization
- Develop skills in multi-task and meta-learning approaches

### Muscle Memory Exercises

#### Exercise Set 1: Advanced Optimization Techniques
**Baseline Code Pattern:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
import math
from typing import Dict, List, Optional, Callable

class WarmupScheduler:
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, 
                 total_steps: int, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr_scale = self.step_count / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
            lr_scale = max(lr_scale, self.min_lr / max(self.base_lrs))
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[i] * lr_scale

class AdamWWithDecoupledWD(optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999), 
                 eps: float = 1e-8, weight_decay: float = 1e-2, amsgrad: bool = False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if group['amsgrad']:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # Decoupled weight decay
                p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                
                # Apply update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss

class GradientClipping:
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def __call__(self, model: nn.Module) -> float:
        return nn.utils.clip_grad_norm_(model.parameters(), self.max_norm, self.norm_type)

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # predictions: [batch_size, num_classes]
        # targets: [batch_size]
        
        num_classes = predictions.size(-1)
        log_probs = F.log_softmax(predictions, dim=-1)
        
        # Smooth targets
        smooth_targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1.0 - self.smoothing
        )
        smooth_targets += self.smoothing / num_classes
        
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
```

**Variations to Practice (Complete all 12):**

1. **Learning Rate Scheduling:**
   - Implement polynomial decay scheduling
   - Add cyclic learning rate with restarts
   - Create adaptive learning rate based on validation loss
   - Design learning rate finder algorithm

2. **Advanced Optimizers:**
   - Implement RAdam (Rectified Adam)
   - Add Lookahead optimizer wrapper
   - Create LAMB optimizer for large batch training
   - Design gradient centralization

3. **Regularization Techniques:**
   - Implement different dropout variants (DropConnect, Scheduled Dropout)
   - Add spectral normalization to layers
   - Create noise-based regularization
   - Design curriculum-based regularization

#### Exercise Set 2: Transfer Learning and Fine-tuning
**Baseline Code Pattern:**
```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Union

class TransferLearningNLP(nn.Module):
    def __init__(self, model_name: str, num_classes: int, 
                 freeze_base: bool = True, dropout_rate: float = 0.1):
        super().__init__()
        
        # Load pretrained model
        self.base_model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.base_model.config.hidden_size
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        # Get base model outputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def unfreeze_layers(self, num_layers: int = 2):
        """Unfreeze top N layers of the base model"""
        layers = list(self.base_model.encoder.layer)
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

class DomainAdaptationTrainer:
    def __init__(self, model: nn.Module, source_loader, target_loader, 
                 lambda_domain: float = 0.1):
        self.model = model
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.lambda_domain = lambda_domain
        
        # Domain discriminator
        self.domain_classifier = nn.Sequential(
            nn.Linear(model.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2)  # Source vs Target
        )
        
    def train_step(self, source_batch, target_batch, optimizer):
        self.model.train()
        
        # Extract features from both domains
        source_features = self.model.base_model(**source_batch).last_hidden_state[:, 0]
        target_features = self.model.base_model(**target_batch).last_hidden_state[:, 0]
        
        # Task loss on source domain
        source_logits = self.model.classifier(source_features)
        task_loss = F.cross_entropy(source_logits, source_batch['labels'])
        
        # Domain adversarial loss
        # Reverse gradient for domain classifier training
        source_domain_logits = self.domain_classifier(self.reverse_gradient(source_features))
        target_domain_logits = self.domain_classifier(self.reverse_gradient(target_features))
        
        source_domain_labels = torch.zeros(len(source_features), dtype=torch.long, 
                                         device=source_features.device)
        target_domain_labels = torch.ones(len(target_features), dtype=torch.long, 
                                        device=target_features.device)
        
        domain_loss = (F.cross_entropy(source_domain_logits, source_domain_labels) + 
                      F.cross_entropy(target_domain_logits, target_domain_labels)) / 2
        
        # Total loss
        total_loss = task_loss + self.lambda_domain * domain_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'task_loss': task_loss.item(),
            'domain_loss': domain_loss.item(),
            'total_loss': total_loss.item()
        }
    
    @staticmethod
    def reverse_gradient(x, alpha=1.0):
        return ReverseLayerF.apply(x, alpha)

class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class ProgressiveUnfreezing:
    def __init__(self, model: nn.Module, total_epochs: int, unfreeze_schedule: List[int]):
        self.model = model
        self.total_epochs = total_epochs
        self.unfreeze_schedule = unfreeze_schedule
        self.current_epoch = 0
        
    def step_epoch(self):
        self.current_epoch += 1
        
        # Check if we should unfreeze more layers
        if self.current_epoch in self.unfreeze_schedule:
            self.unfreeze_next_layer()
    
    def unfreeze_next_layer(self):
        # Find the next layer to unfreeze
        layers = list(self.model.base_model.encoder.layer)
        
        # Count currently unfrozen layers
        unfrozen_count = sum(1 for layer in layers 
                           if any(p.requires_grad for p in layer.parameters()))
        
        # Unfreeze next layer
        if unfrozen_count < len(layers):
            layer_to_unfreeze = layers[-(unfrozen_count + 1)]
            for param in layer_to_unfreeze.parameters():
                param.requires_grad = True
            
            print(f"Unfroze layer {len(layers) - unfrozen_count - 1}")
```

**Variations to Practice (Complete all 10):**

1. **Fine-tuning Strategies:**
   - Implement gradual unfreezing with different schedules
   - Add layer-wise learning rate decay
   - Create task-specific layer adaptation
   - Design discriminative fine-tuning

2. **Domain Adaptation:**
   - Implement different domain adaptation techniques (CORAL, MMD)
   - Add adversarial domain adaptation
   - Create multi-source domain adaptation
   - Design unsupervised domain adaptation

#### Exercise Set 3: Multi-task and Meta-learning
**Baseline Code Pattern:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class MultiTaskModel(nn.Module):
    def __init__(self, shared_encoder: nn.Module, task_heads: Dict[str, nn.Module]):
        super().__init__()
        self.shared_encoder = shared_encoder
        self.task_heads = nn.ModuleDict(task_heads)
        
    def forward(self, x: torch.Tensor, task_name: str):
        # Shared representation
        shared_repr = self.shared_encoder(x)
        
        # Task-specific head
        output = self.task_heads[task_name](shared_repr)
        
        return output

class MultiTaskTrainer:
    def __init__(self, model: MultiTaskModel, task_weights: Dict[str, float] = None):
        self.model = model
        self.task_weights = task_weights or {}
        self.task_losses = defaultdict(list)
        
    def compute_weighted_loss(self, task_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0
        total_weight = 0
        
        for task_name, loss in task_losses.items():
            weight = self.task_weights.get(task_name, 1.0)
            total_loss += weight * loss
            total_weight += weight
        
        return total_loss / total_weight if total_weight > 0 else total_loss
    
    def adaptive_task_weighting(self, task_losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Implement uncertainty weighting for multi-task learning"""
        weights = {}
        
        for task_name, loss in task_losses.items():
            # Store loss for moving average
            self.task_losses[task_name].append(loss.item())
            
            # Keep only recent losses
            if len(self.task_losses[task_name]) > 100:
                self.task_losses[task_name] = self.task_losses[task_name][-100:]
            
            # Compute adaptive weight based on loss variance
            if len(self.task_losses[task_name]) > 10:
                recent_losses = self.task_losses[task_name][-10:]
                variance = torch.var(torch.tensor(recent_losses))
                weights[task_name] = 1.0 / (variance + 1e-8)
            else:
                weights[task_name] = 1.0
        
        return weights

class MAML(nn.Module):
    def __init__(self, model: nn.Module, inner_lr: float = 0.01, meta_lr: float = 0.001):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        
    def inner_loop(self, support_data: Tuple[torch.Tensor, torch.Tensor], 
                   num_steps: int = 1) -> nn.Module:
        """Perform inner loop adaptation"""
        # Create a copy of the model for adaptation
        adapted_model = self.copy_model()
        
        support_x, support_y = support_data
        
        for _ in range(num_steps):
            # Forward pass
            logits = adapted_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, adapted_model.parameters(), 
                                      create_graph=True, retain_graph=True)
            
            # Update parameters
            for param, grad in zip(adapted_model.parameters(), grads):
                param.data = param.data - self.inner_lr * grad
        
        return adapted_model
    
    def meta_update(self, batch_tasks: List[Tuple[Tuple, Tuple]]):
        """Perform meta-learning update"""
        meta_loss = 0
        
        for support_data, query_data in batch_tasks:
            # Inner loop adaptation
            adapted_model = self.inner_loop(support_data)
            
            # Evaluate on query set
            query_x, query_y = query_data
            query_logits = adapted_model(query_x)
            task_loss = F.cross_entropy(query_logits, query_y)
            
            meta_loss += task_loss
        
        # Average across tasks
        meta_loss = meta_loss / len(batch_tasks)
        
        # Meta-optimization step
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def copy_model(self) -> nn.Module:
        """Create a functional copy of the model"""
        # This is a simplified version - in practice, you'd need higher-order optimization
        # libraries like learn2learn or torchmeta for proper implementation
        copied_model = type(self.model)(self.model.config)
        copied_model.load_state_dict(self.model.state_dict())
        return copied_model

class CurriculumLearning:
    def __init__(self, difficulty_fn: callable, initial_difficulty: float = 0.1, 
                 difficulty_step: float = 0.1):
        self.difficulty_fn = difficulty_fn
        self.current_difficulty = initial_difficulty
        self.difficulty_step = difficulty_step
        
    def get_batch(self, dataset, batch_size: int):
        """Get a batch based on current difficulty level"""
        # Filter dataset based on difficulty
        valid_samples = [sample for sample in dataset 
                        if self.difficulty_fn(sample) <= self.current_difficulty]
        
        if len(valid_samples) < batch_size:
            # If not enough samples, increase difficulty
            self.current_difficulty = min(1.0, self.current_difficulty + self.difficulty_step)
            valid_samples = [sample for sample in dataset 
                           if self.difficulty_fn(sample) <= self.current_difficulty]
        
        # Sample batch
        indices = torch.randperm(len(valid_samples))[:batch_size]
        batch = [valid_samples[i] for i in indices]
        
        return batch
    
    def update_difficulty(self, performance_metric: float, threshold: float = 0.8):
        """Update difficulty based on performance"""
        if performance_metric > threshold:
            self.current_difficulty = min(1.0, self.current_difficulty + self.difficulty_step)
        else:
            self.current_difficulty = max(0.1, self.current_difficulty - self.difficulty_step/2)
```

**Variations to Practice (Complete all 8):**

1. **Multi-task Learning:**
   - Implement different sharing strategies (hard, soft, cross-stitch)
   - Add task-specific batch normalization
   - Create hierarchical task relationships
   - Design task clustering for related tasks

2. **Meta-learning:**
   - Implement Prototypical Networks
   - Add Model-Agnostic Meta-Learning variants
   - Create few-shot learning frameworks
   - Design meta-learning for optimization

### Thinking-Heavy Exercises

#### Exercise 1: Training Dynamics Analysis
**Objective:** Understand and analyze training dynamics in NLP models

**Tasks:**
1. **Loss Landscape Analysis:**
   - Visualize loss landscapes during training
   - Analyze mode connectivity between solutions
   - Study the effect of different optimizers on convergence
   - Design training stability metrics

2. **Gradient Analysis:**
   - Track gradient norms and directions during training
   - Analyze gradient noise and its impact on convergence
   - Study gradient conflicts in multi-task learning
   - Design gradient-based early stopping criteria

3. **Generalization Studies:**
   - Analyze overfitting patterns in different model sizes
   - Study the relationship between training and validation dynamics
   - Design generalization-aware training strategies
   - Create robustness metrics for model evaluation

#### Exercise 2: Advanced Regularization Design
**Objective:** Design and implement novel regularization techniques for NLP

**Tasks:**
1. **Information-Theoretic Regularization:**
   - Implement variational information bottleneck
   - Design mutual information regularization
   - Create information-theoretic generalization bounds
   - Build uncertainty quantification methods

2. **Structural Regularization:**
   - Design syntactic structure-aware regularization
   - Implement attention pattern regularization
   - Create semantic consistency regularization
   - Build cross-lingual regularization techniques

3. **Data-Driven Regularization:**
   - Implement adaptive data augmentation
   - Design adversarial training for robustness
   - Create noise injection strategies
   - Build curriculum-based regularization

### Projects

#### Project 2.7: Multi-Domain Sentiment Analysis with Transfer Learning
**Objective:** Build a sentiment analysis system that works across multiple domains using advanced transfer learning

**Requirements:**
- **Data:** Sentiment analysis across domains (movies, products, restaurants, tweets)
- **Architecture:** BERT-based model with domain adaptation layers
- **Features:** Progressive unfreezing, domain adversarial training, uncertainty quantification
- **Evaluation:** Cross-domain generalization, few-shot adaptation, robustness testing

**Key Components:**
1. **Domain Adaptation Framework:**
   - Multi-source domain adaptation architecture
   - Adversarial domain classification for domain-invariant features
   - Progressive unfreezing strategy based on domain similarity
   - Domain-specific fine-tuning with regularization

2. **Advanced Training Strategies:**
   - Curriculum learning from easy to hard domains
   - Meta-learning for fast domain adaptation
   - Uncertainty-aware training with label smoothing
   - Gradient surgery for conflicting domain objectives

3. **Evaluation Framework:**
   - Cross-domain evaluation protocol
   - Few-shot learning on new domains
   - Robustness testing with adversarial examples
   - Interpretability analysis of domain-specific features

4. **System Integration:**
   - Real-time inference with domain detection
   - Confidence estimation for predictions
   - Active learning for new domain adaptation
   - Model versioning and A/B testing framework

#### Project 2.8: Multi-Task Learning for NLP Pipeline
**Objective:** Create a unified multi-task model that performs multiple NLP tasks simultaneously

**Requirements:**
- **Tasks:** Named Entity Recognition, Part-of-Speech Tagging, Dependency Parsing, Sentiment Analysis
- **Architecture:** Shared encoder with task-specific heads and cross-task attention
- **Features:** Dynamic task weighting, hierarchical task relationships, continual learning
- **Evaluation:** Individual task performance, multi-task efficiency, knowledge transfer analysis

**Key Components:**
1. **Multi-Task Architecture:**
   - Shared transformer encoder with task-specific decoders
   - Cross-task attention mechanisms for knowledge sharing
   - Hierarchical task modeling (syntax → semantics)
   - Task embedding for dynamic task conditioning

2. **Training Optimization:**
   - Adaptive task weighting based on learning progress
   - Gradient surgery for conflicting task objectives
   - Task scheduling and curriculum learning
   - Multi-task batch sampling strategies

3. **Continual Learning:**
   - Elastic weight consolidation for task sequence learning
   - Progressive neural networks for task expansion
   - Memory replay mechanisms for old task retention
   - Catastrophic forgetting prevention strategies

4. **Analysis and Interpretation:**
   - Task relationship visualization and analysis
   - Knowledge transfer quantification between tasks
   - Task-specific vs shared representation analysis
   - Ablation studies on architecture components

#### Project 2.9: Few-Shot Learning for Low-Resource NLP
**Objective:** Develop a few-shot learning system for NLP tasks in low-resource languages or domains

**Requirements:**
- **Data:** Few-shot datasets for multiple NLP tasks and languages
- **Architecture:** Meta-learning framework with prototypical networks and MAML
- **Features:** Cross-lingual transfer, data augmentation, active learning
- **Evaluation:** Few-shot performance, zero-shot transfer, sample efficiency

**Key Components:**
1. **Meta-Learning Framework:**
   - Model-Agnostic Meta-Learning (MAML) implementation
   - Prototypical networks for similarity-based classification
   - Matching networks with attention mechanisms
   - Meta-learning with memory-augmented networks

2. **Cross-Lingual Transfer:**
   - Multilingual pre-trained model fine-tuning
   - Cross-lingual alignment techniques
   - Language-agnostic representation learning
   - Zero-shot cross-lingual transfer evaluation

3. **Data Efficiency Techniques:**
   - Intelligent data augmentation for few-shot scenarios
   - Active learning for optimal sample selection
   - Semi-supervised learning with unlabeled data
   - Self-training and pseudo-labeling strategies

4. **Evaluation and Analysis:**
   - Few-shot learning curves and sample efficiency
   - Cross-lingual transfer effectiveness analysis
   - Robustness testing across different domains
   - Meta-learning convergence and stability analysis

## Assessment and Milestones

### Week 8 Checkpoint: Text Processing Mastery
**Technical Assessment:**
- Complete implementation of word embedding algorithms (Word2Vec, GloVe, FastText)
- Demonstrate proficiency in text preprocessing pipelines
- Build attention mechanism from scratch with visualization

**Portfolio Requirements:**
- Text classification system with custom embeddings
- Attention visualization tool with interactive interface
- Comprehensive tokenization benchmark across languages

### Week 10 Checkpoint: Sequence Modeling Expertise  
**Technical Assessment:**
- Implement custom RNN variants with mathematical understanding
- Build sequence-to-sequence model with attention from scratch
- Demonstrate bidirectional and hierarchical modeling capabilities

**Portfolio Requirements:**
- Neural machine translation system with custom architecture
- Dialogue system with conversational memory
- Text summarization with extractive and abstractive components

### Week 12 Checkpoint: Advanced Training Techniques
**Technical Assessment:**
- Implement meta-learning algorithm for few-shot NLP
- Demonstrate multi-task learning with dynamic task weighting
- Build domain adaptation system with adversarial training

**Portfolio Requirements:**
- Multi-domain sentiment analysis with transfer learning
- Multi-task NLP pipeline with shared representations
- Few-shot learning system for low-resource scenarios

### Final Phase 2 Portfolio
**Comprehensive System Integration:**
- Production-ready NLP pipeline with multiple components
- Advanced training framework with modern optimization
- Research-quality implementations with proper evaluation

**Research Contribution:**
- Novel combination of techniques from different papers
- Thorough ablation studies and analysis
- Clear documentation and reproducible results
- Potential for publication or open-source contribution

## Success Metrics

### Technical Competencies
- **Implementation Mastery:** Ability to implement complex NLP architectures from scratch
- **Training Expertise:** Proficiency with advanced optimization and regularization techniques  
- **Transfer Learning:** Skill in adapting models across domains and tasks
- **Research Skills:** Capability to reproduce and extend research papers

### Portfolio Quality
- **Code Quality:** Professional, well-documented, and maintainable implementations
- **Experimental Rigor:** Proper evaluation protocols and statistical analysis
- **Innovation:** Creative combinations and extensions of existing techniques
- **Communication:** Clear presentation of technical concepts and results

### Industry Readiness
- **Production Skills:** Experience with deployment and optimization considerations
- **Problem Solving:** Ability to design solutions for real-world NLP challenges
- **Research Awareness:** Understanding of current trends and state-of-the-art
- **Collaboration:** Skills in version control, documentation, and knowledge sharing