import state



GRAPH_PREP_START_MSG = """
============================
STARTING GRAPH CONFIGURATION
============================
"""


AVAIL_ATOM_FEATURES = """
Studied Features Set:
---------------------
- Atomic Number
- Aromaticity
- Number of Hydrogens

All Available Features:
---------------------
- Atomic Number
- Atom Charge
- Aromaticity
- Number of Hydrogens
- Atom Hybridization
- Atomic Mass
- Atom Valence
- Atom Degree

"""



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
      
Welcome to GNN LAB (R), version {state.VERSION}. The scientific programm for training molecular GNN models to predict desired properties.      
Available MPNN models architectures:

- Graph Convolution Networks (GCN)
- Graph Isomorphism Networks (GIN)
- Graph Attention Networks (GAT)
      
"""