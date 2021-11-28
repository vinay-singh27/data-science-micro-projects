import torch
import torch.nn as nn


class FarePredictionModel(nn.Module) :
    
    def __init__(self, embedding_size, num_of_numeric_feats, output_size, layers, p =0.5) :
        
        super().__init__()
        
        #creating embedding module
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in embedding_size])
        self.embeds_dropout = nn.Dropout(p)
        
        #batch normalization for numeric features
        self.batch_norm = nn.BatchNorm1d(num_of_numeric_feats)
        
        #define input layer
        num_of_embed_layer = sum([nf for ni,nf in embedding_size])
        input_size = num_of_embed_layer + num_of_numeric_feats
        
        #defining layers
        layer_list = []
        
        for i in layers :
            layer_list.append(nn.Linear(in_features= input_size, out_features= i))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.BatchNorm1d(i))
            layer_list.append(nn.Dropout(p))
            input_size = i
            
        #appened outer layer
        layer_list.append(nn.Linear(layers[-1], output_size))
        
        #compile all the layers
        self.layers = nn.Sequential(*layer_list)
        
        
    def forward(self, X_Categorical, X_Numerical) :
        
        #embedding for categorical data
        embeddings_data = []
        for i,e in enumerate(self.embeds) :
            embeddings_data.append(e(X_Categorical[: ,i]))
        
            
        X = torch.cat(embeddings_data, 1)
        X = self.embeds_dropout(X)
        
        X_Numerical = self.batch_norm(X_Numerical)
        
        X = torch.cat([X, X_Numerical], axis=1)
        X = self.layers(X)
        
        return X


def train_test_split(categorical_data, numerical_data,y,
                     batch_size, test_ratio) :

    #split into train & test
    test_size = int(batch_size * test_ratio)

    cat_train = categorical_data[:batch_size-test_size]
    cat_test = categorical_data[batch_size-test_size:batch_size]
    con_train = numerical_data[:batch_size-test_size]
    con_test = numerical_data[batch_size-test_size:batch_size]
    y_train = y[:batch_size-test_size]
    y_test = y[batch_size-test_size:batch_size]

    return cat_train, cat_test, con_train, con_test, y_train, y_test
        
