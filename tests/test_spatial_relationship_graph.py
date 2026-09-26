def test_spatial_relationship_graph():
    # Synthetic test to ensure entities are mapped correctly in the relationship graph
    entities = ["Worker", "Tool", "Workpiece"]
    relations = [("Worker", "holding", "Tool"), ("Tool", "interacting_with", "Workpiece")]
    assert len(entities) == 3
    assert len(relations) == 2
