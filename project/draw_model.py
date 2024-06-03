from graphviz import Digraph


def plot_bcmodel():
    dot = Digraph(format='png')

    # Define the nodes
    dot.node('X1', 'Input x1 (T, C1, H, W)')
    dot.node('X2', 'Input x2 (T, C2)')
    dot.node('ENC', 'Encoder1')
    dot.node('SAW', 'SpatialAttentionWeight')
    dot.node('MID', 'MidIncepNet')
    dot.node('GAP', 'GAP')
    dot.node('TDVE', 'TimeDistributedValueEncoder')
    dot.node('CONC', 'Concatenate')
    dot.node('DEC', 'CustomDecoder')
    dot.node('OUT', 'Corrected Output')

    # Define the edges
    dot.edges([
        ('X1', 'ENC'),
        ('ENC', 'SAW'),
        ('SAW', 'MID'),
        ('MID', 'GAP'),
        ('X2', 'TDVE'),
        ('GAP', 'CONC'),
        ('TDVE', 'CONC'),
        ('CONC', 'DEC'),
        ('DEC', 'OUT')
    ])

    # Render and save the diagram
    dot.render('bcmodel_architecture')
    return dot


# Plot the model
bcmodel_diagram = plot_bcmodel()
bcmodel_diagram.view()