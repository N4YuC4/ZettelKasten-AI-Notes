# tests/test_mind_map_widget.py
import pytest
import math
from unittest.mock import MagicMock
from mind_map_widget import MindMapWidget

def test_mind_map_invalidates_cache_on_title_rename():
    """
    Verifies FLAW-MM-01: Layout cache recomputes bounding box when a note is renamed.
    Mocks canvas.update to prevent unmounted Flet Canvas runtime errors.
    """
    widget = MindMapWidget(None, None)
    widget.canvas.update = MagicMock()
    
    widget.update_map([("id-1", "Short", "")], [])
    old_size = widget.notes["id-1"]["size"]
    
    widget.update_map([("id-1", "A Very Long New Note Title After Rename", "")], [])
    new_size = widget.notes["id-1"]["size"]
    
    assert new_size[0] > old_size[0]
    assert widget.canvas.update.called

def test_mind_map_fallback_grid_spacing_prevents_overlap():
    """
    Verifies FLAW-MM-02: Fallback grid separates wide nodes without bounding box collisions.
    Mocks canvas.update to prevent unmounted Flet Canvas runtime errors.
    """
    widget = MindMapWidget(None, None)
    widget.canvas.update = MagicMock()
    
    notes = [
        ("1", "Very Long Note Title Number One", ""),
        ("2", "Very Long Note Title Number Two", "")
    ]
    widget.update_map(notes, [])
    rect1 = widget.notes["1"]["rect"]
    rect2 = widget.notes["2"]["rect"]
    
    # Verify rect1 and rect2 do not overlap on the x-axis if placed in the same row
    if rect1[1] == rect2[1]:
        assert rect1[2] <= rect2[0] or rect2[2] <= rect1[0]
    assert widget.canvas.update.called

def test_mind_map_disconnected_clusters_no_overlap():
    """
    Verifies that multiple disconnected clusters and isolated nodes do not overlap.
    """
    widget = MindMapWidget(None, None)
    widget.canvas.update = MagicMock()
    
    notes = [
        ("c1_a", "Cluster 1 Node A", ""),
        ("c1_b", "Cluster 1 Node B", ""),
        ("c2_a", "Cluster 2 Node A", ""),
        ("c2_b", "Cluster 2 Node B", ""),
        ("iso_1", "Isolated Node Alpha", ""),
        ("iso_2", "Isolated Node Beta", "")
    ]
    links = [
        ("c1_a", "c1_b"),
        ("c2_a", "c2_b")
    ]
    widget.update_map(notes, links)
    
    # Check all pairs for bounding box collisions
    note_ids = list(widget.notes.keys())
    for i in range(len(note_ids)):
        for j in range(i + 1, len(note_ids)):
            r1 = widget.notes[note_ids[i]]["rect"]
            r2 = widget.notes[note_ids[j]]["rect"]
            # Two boxes overlap if and only if they intersect on both X and Y
            x_overlap = (r1[0] < r2[2]) and (r1[2] > r2[0])
            y_overlap = (r1[1] < r2[3]) and (r1[3] > r2[1])
            assert not (x_overlap and y_overlap), f"Overlap detected between {note_ids[i]} and {note_ids[j]}"

def test_mind_map_selection_callback_invoked():
    """
    Verifies FLAW-MM-03: Tapping a node bounding box invokes on_note_selected callback.
    """
    selected_ids = []
    widget = MindMapWidget(None, on_note_selected=lambda nid: selected_ids.append(nid))
    widget.canvas.update = MagicMock()
    
    widget.update_map([("id-100", "Node Target", "")], [])
    rect = widget.notes["id-100"]["rect"]
    
    # Simulate tap inside bounding box
    mid_x = (rect[0] + rect[2]) / 2
    mid_y = (rect[1] + rect[3]) / 2
    
    tap_event = MagicMock()
    tap_event.local_position.x = mid_x
    tap_event.local_position.y = mid_y
    
    widget.handle_tap_down(tap_event)
    assert "id-100" in selected_ids

def test_mind_map_tap_fallback_coordinates():
    """
    Verifies tap down handler gracefully supports local_x/y and global_position attributes.
    """
    selected_ids = []
    widget = MindMapWidget(None, on_note_selected=lambda nid: selected_ids.append(nid))
    widget.canvas.update = MagicMock()
    
    widget.update_map([("id-200", "Fallback Target", "")], [])
    rect = widget.notes["id-200"]["rect"]
    mid_x = (rect[0] + rect[2]) / 2
    mid_y = (rect[1] + rect[3]) / 2
    
    # Fallback to local_x, local_y
    tap_event = MagicMock(spec=["local_x", "local_y"])
    tap_event.local_x = mid_x
    tap_event.local_y = mid_y
    
    widget.handle_tap_down(tap_event)
    assert "id-200" in selected_ids

def test_mind_map_explicit_invalidate_cache():
    """
    Verifies explicit invalidate_cache resets cached positions and fingerprints.
    """
    widget = MindMapWidget(None, None)
    widget.canvas.update = MagicMock()
    widget.update_map([("1", "Title 1", "")], [])
    assert len(widget.node_positions) == 1
    
    widget.invalidate_cache()
    assert len(widget.node_positions) == 0
    assert widget._last_nodes_fingerprint is None

def test_mind_map_empty_notes_handled_cleanly():
    """
    Verifies empty note set resets shapes and handles cleanly.
    """
    widget = MindMapWidget(None, None)
    widget.canvas.update = MagicMock()
    widget.update_map([], [])
    assert len(widget.notes) == 0
    assert len(widget.canvas.shapes) == 0
