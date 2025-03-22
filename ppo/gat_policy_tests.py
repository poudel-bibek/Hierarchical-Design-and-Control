import torch
from torch_geometric.data import Data, Batch
from models import GAT_v2_ActorCritic

def test_model():
    # Create model
    in_channels = 2
    action_dim = 5
    kwargs = {
        'hidden_channels': 16, 'out_channels': 32, 'initial_heads': 2,
        'second_heads': 1, 'edge_dim': 3, 'min_thickness': 0.1,
        'max_thickness': 10.0, 'num_mixtures': 3, 'activation': 'relu',
        'model_size': 'small'
    }
    model = GAT_v2_ActorCritic(in_channels, action_dim, **kwargs)

    # Create mock data
    x = torch.randn(10, in_channels)
    edge_index = torch.tensor([[0,1,1,2,2,3,3,4,4,0], [1,0,2,1,3,2,4,3,0,4]], dtype=torch.long)
    edge_attr = torch.randn(edge_index.shape[1], kwargs['edge_dim'])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Debug print
    print(f"Creating batch from {len([data, data])} graphs")
    batch = Batch.from_data_list([data, data])  # Create a batch of 2 graphs
    print(f"Batch created with num_graphs: {batch.num_graphs}")

    # Test 3: Action sampling
    proposals, counts, log_probs = model.act(batch, training=True, iteration=1)
    print(f"After act - Proposals shape: {proposals.shape}")


    # Test 4: Bounds checking
    valid_locations = (proposals == -1) | ((proposals >= 0) & (proposals <= 1))
    valid_thickness = (proposals == -1) | ((proposals >= kwargs['min_thickness']) &
                                          (proposals <= kwargs['max_thickness']))
    print(f"Valid locations: {valid_locations[:,:,0].all()}")
    print(f"Valid thickness: {valid_thickness[:,:,1].all()}")

    # Test 5: Evaluation
    action_log_probs, state_values, entropy = model.evaluate(batch, proposals)
    print(f"Eval results: log_probs={action_log_probs.shape}, "
          f"values={state_values.shape}, entropy={entropy.shape}")

    # Test 6: Parameter count
    param_counts = model.param_count()
    print(f"Total params: {param_counts['Grand total']}")

    return "All tests completed!"

test_model()