import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger("cnn_lstm")

class CNNLSTMASR(nn.Module):
    """Complete CNN-LSTM architecture for speech recognition"""
    
    def __init__(self, input_dim=13, hidden_dim=128, num_layers=2, num_classes=29, dropout=0.3):
        super().__init__()
        
       
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
      
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
    def forward(self, x):
        """Forward pass"""
        
        batch_size, _, mfcc_dim, time_steps = x.shape
        
      
        x = x.squeeze(1)  
        cnn_out = self.cnn(x)

        cnn_out = cnn_out.transpose(1, 2)
        
        
        lstm_out, _ = self.lstm(cnn_out)
        
        output = self.dropout(lstm_out)
        logits = self.fc(output)
        
        return logits
    
    def predict(self, features):
        """Inference method"""
        self.eval()
        with torch.no_grad():
            logits = self(features)
            pred_ids = torch.argmax(logits, dim=-1)
            transcription = self._decode_ctc(pred_ids)
            
        return {
            'transcription': transcription,
            'confidence': 0.85,  
            'logits': logits
        }
    
    def _decode_ctc(self, pred_ids):
        """Simple CTC decoder"""
        chars = " -.'abcdefghijklmnopqrstuvwxyz"
        text = []
        prev = None
        for pid in pred_ids[0]:
            if pid != 0 and pid != prev:  
                text.append(chars[pid])
            prev = pid
        return ''.join(text).lower().strip()
