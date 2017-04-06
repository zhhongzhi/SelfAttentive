import torch
import torch.nn as nn
from torch.autograd import Variable
from data import *
from model import *
from Utility import *

cuda = True
path = 'data/yelp/'
test_batch_size = 32

# Define the Loss Function for Training
entropy_loss = nn.CrossEntropyLoss()
if cuda:
    entropy_loss.cuda()

# Loading Test Data
corpus = Corpus( path )
print 'Dataset loaded.'
    
# Make test data batchifiable
test_data = select_data( corpus.test, test_batch_size )
test_len = select_data( corpus.test_len, test_batch_size )
test_label = select_data( corpus.test_label, test_batch_size )
    
# Load Trained Model
model = torch.load( 'Attentive-yelp.pt' )
    
# Calculate and Save weights
Weights = Attentive_weights( model, test_data, test_label, test_len, test_batch_size, cuda )

torch.save( Weights, 'attention_weights.pt' )