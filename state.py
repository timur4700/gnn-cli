from app_state import AppState

VERSION = '0.1.0'

APP_STATE: AppState


ATOM_CODES = atom_codes = {"O": 0, 
                           "Cl": 1, 
                           "H": 2, 
                           "N": 3, 
                           "Se": 4, 
                           "I": 5, 
                           "Br": 6, 
                           "S": 7, 
                           "P": 8, 
                           "C": 9, 
                           "F": 10}



INTRO = f"""
╔═════════════════════════════════╗
║                                 ║
║   ██████╗ ███╗   ██╗███╗   ██╗  ║
║  ██╔════╝ ████╗  ██║████╗  ██║  ║
║  ██║  ███╗██╔██╗ ██║██╔██╗ ██║  ║
║  ██║   ██║██║╚██╗██║██║╚██╗██║  ║
║  ╚██████╔╝██║ ╚████║██║ ╚████║  ║
║   ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝  ║
║                                 ║
╚═════════════════════════════════╝
      
Welcome to GNN LAB (R), version {VERSION}. The scientific programm for training molecular GNN models to predict desired properties.      
Available MPNN models architectures:

- Graph Convolution Networks (GCN)
- Graph Isomorphism Networks (GIN)
- Graph Attention Networks (GAT)
      
"""