from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QRect, QTimer
import math
from logger import log_debug

class MindMapWidget(QWidget):
    note_selected = pyqtSignal(str) # Emits the ID of the selected note

    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.notes = {}  # {note_id: {'title': title, 'pos': QPointF, 'rect': QRectF}}
        self.links = []  # [(source_id, target_id)]
        self.current_note_id = None
        from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QRect, QTimer
import math
from logger import log_debug

class MindMapWidget(QWidget):
    note_selected = pyqtSignal(str) # Emits the ID of the selected note

    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.notes = {}  # {note_id: {'title': title, 'pos': QPointF, 'rect': QRectF}}
        self.links = []  # [(source_id, target_id)]
        self.current_note_id = None
        self.node_radius = 10 # Default size
        self.node_padding = 5 # Internal padding for text within node
        self.node_margin = 40 # External spacing between connected nodes
        self.isolated_node_margin = 120 # External spacing for isolated nodes
        self.vertical_spacing_factor = 2.5 # Factor to increase vertical spacing
        self.font = QFont("Arial", 10)
        self.zoom_factor = 1.0 # Keep this one
        self.offset_x = 0.0 # New: X offset for panning
        self.offset_y = 0.0 # New: Y offset for panning
        
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True) # Enable mouse tracking for hover effects

        # Debounce timer for resize events
        self.layout_timer = QTimer(self)
        self.layout_timer.setSingleShot(True)
        self.layout_timer.timeout.connect(self._perform_layout)
        self.layout_delay_ms = 100 # milliseconds

    def update_map(self, all_notes_metadata, all_links, current_note_id=None):
        # log_debug(f"DEBUG: MindMapWidget.update_map called. Notes metadata count: {len(all_notes_metadata)}, Links count: {len(all_links)}, Current note ID: {current_note_id}")
        self.notes.clear()
        self.links = all_links
        self.current_note_id = current_note_id

        # Populate notes dictionary with initial data
        for note_id, title, _ in all_notes_metadata:
            self.notes[note_id] = {'title': title, 'pos': QPointF(), 'rect': QRectF()}

        self._layout_nodes()
        self.repaint() # Force a repaint

    def _perform_layout(self):
        """Performs the layout and updates the widget."""
        self._layout_nodes()
        self.update()

    def _layout_nodes(self):
        # log_debug(f"DEBUG: MindMapWidget._layout_nodes called. Number of notes: {len(self.notes)}")
        if not self.notes:
            # log_debug("DEBUG: MindMapWidget._layout_nodes: No notes to layout.")
            return

        center_x = self.width() / 2
        center_y = self.height() / 2

        num_nodes = len(self.notes)
        if num_nodes == 1:
            note_id = list(self.notes.keys())[0]
            self.notes[note_id]['pos'] = QPointF(center_x, center_y)
            # log_debug(f"DEBUG: MindMapWidget._layout_nodes: Single note layout at {center_x}, {center_y}")
            return

        # 1. Pre-calculate node dimensions and store them
        for note_id, data in self.notes.items():
            text_rect = self.fontMetrics().boundingRect(QRect(0, 0, 1000, 1000), Qt.AlignCenter, data['title'])
            node_width = max(self.node_radius * 2, text_rect.width() + self.node_padding * 2)
            node_height = max(self.node_radius * 2, text_rect.height() + self.node_padding * 2)
            self.notes[note_id]['size'] = (node_width, node_height)

        # 2. Build adjacency list for graph traversal
        adj = {note_id: [] for note_id in self.notes}
        rev_adj = {note_id: [] for note_id in self.notes} # For finding roots
        for source_id, target_id in self.links:
            if source_id in adj and target_id in adj:
                adj[source_id].append(target_id)
                rev_adj[target_id].append(source_id)

        # 3. Identify root nodes (nodes with no incoming links, or current_note_id)
        roots = []
        if self.current_note_id and self.current_note_id in self.notes:
            roots.append(self.current_note_id)
        else:
            for note_id in self.notes:
                if not rev_adj[note_id]: # No incoming links
                    roots.append(note_id)
            if not roots and self.notes: # If there are cycles, pick an arbitrary node
                roots.append(list(self.notes.keys())[0])
        
        # Use a set to keep track of visited nodes to avoid infinite loops in cyclic graphs
        visited = set()
        
        # 4. Assign levels and horizontal order using BFS/DFS
        levels = {} # {note_id: level}
        level_nodes = {} # {level: [note_id, ...]}
        
        queue = [(root_id, 0) for root_id in roots]
        for root_id in roots:
            levels[root_id] = 0
            if 0 not in level_nodes:
                level_nodes[0] = []
            level_nodes[0].append(root_id)
            visited.add(root_id)

        head = 0
        while head < len(queue):
            current_node_id, level = queue[head]
            head += 1

            for neighbor_id in adj[current_node_id]:
                if neighbor_id not in visited:
                    levels[neighbor_id] = level + 1
                    if level + 1 not in level_nodes:
                        level_nodes[level + 1] = []
                    level_nodes[level + 1].append(neighbor_id)
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, level + 1))
                elif levels[neighbor_id] > level + 1: # Found a shorter path to an already visited node
                    levels[neighbor_id] = level + 1
                    # Re-add to queue to re-process its children if its level changed
                    queue.append((neighbor_id, level + 1))

        # Handle unvisited nodes (isolated nodes or separate components)
        unvisited_nodes = [note_id for note_id in self.notes if note_id not in visited]
        if unvisited_nodes:
            # Assign unvisited nodes to a new "level" for layout
            # We'll place them below the main graph or to the side
            
            # Determine the starting level for unvisited nodes
            start_level_for_unvisited = (max(levels.values()) + 2) if levels else 0
            
            # Sort unvisited nodes for deterministic placement
            unvisited_nodes.sort()

            nodes_per_row = int(math.sqrt(len(unvisited_nodes))) + 1 # Simple grid
            
            for i, note_id in enumerate(unvisited_nodes):
                row = i // nodes_per_row
                col = i % nodes_per_row
                
                # Create a virtual level for these nodes
                virtual_level = start_level_for_unvisited + row
                if virtual_level not in level_nodes:
                    level_nodes[virtual_level] = []
                level_nodes[virtual_level].append(note_id)
                levels[note_id] = virtual_level

        # Sort nodes within each level for consistent horizontal placement
        for level in level_nodes:
            level_nodes[level].sort() # Sort by note_id for deterministic order

        # 5. Calculate positions
        max_level = max(levels.values()) if levels else 0
        
        # Vertical spacing
        y_step = (self.height() / (max_level + 2) if max_level > 0 else self.height() / 2) * self.vertical_spacing_factor
        
        # Horizontal spacing and positioning
        for level in sorted(level_nodes.keys()):
            nodes_at_level = level_nodes[level]
            
            # Determine which margin to use for this level
            current_margin = self.node_margin
            # Check if this level contains nodes that were originally unvisited
            # This assumes 'start_level_for_unvisited' was set in the unvisited_nodes handling
            # and that levels for unvisited nodes are distinct and higher.
            # A more robust check might involve checking if all nodes in level_nodes[level] are in unvisited_nodes
            # For now, let's assume levels for unvisited nodes are higher than connected nodes.
            if 'start_level_for_unvisited' in locals() and level >= start_level_for_unvisited:
                current_margin = self.isolated_node_margin

            # Calculate total width required for nodes at this level, including margins
            total_level_width = 0
            for i, node_id in enumerate(nodes_at_level):
                node_width, _ = self.notes[node_id]['size']
                total_level_width += node_width
                if i < len(nodes_at_level) - 1:
                    total_level_width += current_margin # Margin between nodes

            # Start x position to center the level
            current_x = (self.width() - total_level_width) / 2
            if current_x < current_margin: # Ensure it doesn't go too far left
                current_x = current_margin

            for node_id in nodes_at_level:
                node_width, node_height = self.notes[node_id]['size']
                
                x = current_x + node_width / 2
                y = (level + 1) * y_step # +1 to avoid placing at very top edge
                self.notes[node_id]['pos'] = QPointF(x, y)
                
                current_x += node_width + current_margin # Move for next node, adding margin

        # log_debug(f"DEBUG: MindMapWidget._layout_nodes: Layered tree layout complete.")

    def paintEvent(self, event):
        # log_debug(f"DEBUG: MindMapWidget.paintEvent called. Widget size: {self.width()}x{self.height()}")
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self.font)
        # Correct order: scale then translate
        painter.scale(self.zoom_factor, self.zoom_factor)
        painter.translate(self.offset_x, self.offset_y)

        # Draw links first
        for source_id, target_id in self.links:
            if source_id in self.notes and target_id in self.notes:
                start_pos = self.notes[source_id]['pos']
                end_pos = self.notes[target_id]['pos']
                # log_debug(f"DEBUG: MindMapWidget.paintEvent: Drawing link from {source_id} to {target_id}")

                # Draw arrow
                painter.setPen(QPen(QColor(0, 0, 255), 2)) # Blue pen, increased thickness
                painter.drawLine(start_pos, end_pos)

                # Simple arrow head (optional, can be improved)
                # dx = end_pos.x() - start_pos.x()
                # dy = end_pos.y() - end_pos.y()
                # angle = math.atan2(dy, dx)
                # arrow_size = 8
                # painter.drawLine(end_pos, end_pos - QPointF(arrow_size * math.cos(angle - math.pi / 6), arrow_size * math.sin(angle - math.pi / 6)))
                # painter.drawLine(end_pos, end_pos - QPointF(arrow_size + math.cos(angle + math.pi / 6), arrow_size * math.sin(angle + math.pi / 6)))


        # Draw nodes
        for note_id, data in self.notes.items():
            pos = data['pos']
            title = data['title']
            # log_debug(f"DEBUG: MindMapWidget.paintEvent: Drawing node {note_id} with title '{title}' at {pos.x()}, {pos.y()}")

            node_width, node_height = data['size']

            # Adjust rect position to be centered around pos
            rect = QRectF(pos.x() - node_width / 2, pos.y() - node_height / 2, node_width, node_height)
            self.notes[note_id]['rect'] = rect # Store for click detection

            # Node background
            if note_id == self.current_note_id:
                painter.setBrush(QColor(0, 255, 0)) # Bright green for highlighted
            else:
                painter.setBrush(QColor(255, 0, 0)) # Bright red for default
            painter.setPen(QPen(QColor(0, 0, 0), 3))
            painter.drawRoundedRect(rect, 10, 10) # Rounded rectangle

            # Node text
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(rect, Qt.AlignCenter, title)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Transform mouse position from widget coordinates to logical coordinates
            # Reverse the transformations applied in paintEvent
            # 1. Translate by -offset_x, -offset_y
            # 2. Scale by 1 / zoom_factor
            transformed_x = (event.pos().x() / self.zoom_factor) - self.offset_x
            transformed_y = (event.pos().y() / self.zoom_factor) - self.offset_y
            transformed_pos = QPointF(transformed_x, transformed_y)

            for note_id, data in self.notes.items():
                if data['rect'].contains(transformed_pos):
                    self.current_note_id = note_id
                    self.note_selected.emit(note_id)
                    self.update()
                    break
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        zoom_in_factor = 1.1
        zoom_out_factor = 0.9

        old_zoom_factor = self.zoom_factor
        if event.angleDelta().y() > 0: # Scrolled up (zoom in)
            self.zoom_factor *= zoom_in_factor
        else: # Scrolled down (zoom out)
            self.zoom_factor *= zoom_out_factor

        # Clamp zoom factor
        self.zoom_factor = max(0.1, min(self.zoom_factor, 5.0))

        # Adjust position to zoom around mouse cursor
        # Get mouse position relative to the widget
        mouse_pos = event.pos()

        # Correct formula when translate is applied AFTER scale
        self.offset_x = (mouse_pos.x() / self.zoom_factor) - (mouse_pos.x() / old_zoom_factor) + self.offset_x
        self.offset_y = (mouse_pos.y() / self.zoom_factor) - (mouse_pos.y() / old_zoom_factor) + self.offset_y

        self.update()
        super().wheelEvent(event)

    def resizeEvent(self, event):
        # log_debug(f"DEBUG: MindMapWidget.resizeEvent called. New size: {event.size().width()}x{event.size().height()}")
        self.layout_timer.stop()
        self.layout_timer.start(self.layout_delay_ms)
        super().resizeEvent(event)

        self.zoom_factor = 1.0 # Keep this one
        self.offset_x = 0.0 # New: X offset for panning
        self.offset_y = 0.0 # New: Y offset for panning
        
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True) # Enable mouse tracking for hover effects

        # Debounce timer for resize events
        self.layout_timer = QTimer(self)
        self.layout_timer.setSingleShot(True)
        self.layout_timer.timeout.connect(self._perform_layout)
        self.layout_delay_ms = 100 # milliseconds

    def update_map(self, all_notes_metadata, all_links, current_note_id=None):
        # log_debug(f"DEBUG: MindMapWidget.update_map called. Notes metadata count: {len(all_notes_metadata)}, Links count: {len(all_links)}, Current note ID: {current_note_id}")
        self.notes.clear()
        self.links = all_links
        self.current_note_id = current_note_id

        # Populate notes dictionary with initial data
        for note_id, title, _ in all_notes_metadata:
            self.notes[note_id] = {'title': title, 'pos': QPointF(), 'rect': QRectF()}

        self._layout_nodes()
        self.repaint() # Force a repaint

    def _perform_layout(self):
        """Performs the layout and updates the widget."""
        self._layout_nodes()
        self.update()

    def _layout_nodes(self):
        # log_debug(f"DEBUG: MindMapWidget._layout_nodes called. Number of notes: {len(self.notes)}")
        if not self.notes:
            # log_debug("DEBUG: MindMapWidget._layout_nodes: No notes to layout.")
            return

        center_x = self.width() / 2
        center_y = self.height() / 2

        num_nodes = len(self.notes)
        if num_nodes == 1:
            note_id = list(self.notes.keys())[0]
            self.notes[note_id]['pos'] = QPointF(center_x, center_y)
            # log_debug(f"DEBUG: MindMapWidget._layout_nodes: Single note layout at {center_x}, {center_y}")
            return

        # 1. Pre-calculate node dimensions and store them
        for note_id, data in self.notes.items():
            text_rect = self.fontMetrics().boundingRect(QRect(0, 0, 1000, 1000), Qt.AlignCenter, data['title'])
            node_width = max(self.node_radius * 2, text_rect.width() + self.node_padding * 2)
            node_height = max(self.node_radius * 2, text_rect.height() + self.node_padding * 2)
            self.notes[note_id]['size'] = (node_width, node_height)

        # 2. Build adjacency list for graph traversal
        adj = {note_id: [] for note_id in self.notes}
        rev_adj = {note_id: [] for note_id in self.notes} # For finding roots
        for source_id, target_id in self.links:
            if source_id in adj and target_id in adj:
                adj[source_id].append(target_id)
                rev_adj[target_id].append(source_id)

        # 3. Identify root nodes (nodes with no incoming links, or current_note_id)
        roots = []
        if self.current_note_id and self.current_note_id in self.notes:
            roots.append(self.current_note_id)
        else:
            for note_id in self.notes:
                if not rev_adj[note_id]: # No incoming links
                    roots.append(note_id)
            if not roots and self.notes: # If there are cycles, pick an arbitrary node
                roots.append(list(self.notes.keys())[0])
        
        # Use a set to keep track of visited nodes to avoid infinite loops in cyclic graphs
        visited = set()
        
        # 4. Assign levels and horizontal order using BFS/DFS
        levels = {} # {note_id: level}
        level_nodes = {} # {level: [note_id, ...]}
        
        queue = [(root_id, 0) for root_id in roots]
        for root_id in roots:
            levels[root_id] = 0
            if 0 not in level_nodes:
                level_nodes[0] = []
            level_nodes[0].append(root_id)
            visited.add(root_id)

        head = 0
        while head < len(queue):
            current_node_id, level = queue[head]
            head += 1

            for neighbor_id in adj[current_node_id]:
                if neighbor_id not in visited:
                    levels[neighbor_id] = level + 1
                    if level + 1 not in level_nodes:
                        level_nodes[level + 1] = []
                    level_nodes[level + 1].append(neighbor_id)
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, level + 1))
                elif levels[neighbor_id] > level + 1: # Found a shorter path to an already visited node
                    levels[neighbor_id] = level + 1
                    # Re-add to queue to re-process its children if its level changed
                    queue.append((neighbor_id, level + 1))

        # Handle unvisited nodes (isolated nodes or separate components)
        unvisited_nodes = [note_id for note_id in self.notes if note_id not in visited]
        if unvisited_nodes:
            # Assign unvisited nodes to a new "level" for layout
            # We'll place them below the main graph or to the side
            
            # Determine the starting level for unvisited nodes
            start_level_for_unvisited = (max(levels.values()) + 2) if levels else 0
            
            # Sort unvisited nodes for deterministic placement
            unvisited_nodes.sort()

            nodes_per_row = int(math.sqrt(len(unvisited_nodes))) + 1 # Simple grid
            
            for i, note_id in enumerate(unvisited_nodes):
                row = i // nodes_per_row
                col = i % nodes_per_row
                
                # Create a virtual level for these nodes
                virtual_level = start_level_for_unvisited + row
                if virtual_level not in level_nodes:
                    level_nodes[virtual_level] = []
                level_nodes[virtual_level].append(note_id)
                levels[note_id] = virtual_level

        # Sort nodes within each level for consistent horizontal placement
        for level in level_nodes:
            level_nodes[level].sort() # Sort by note_id for deterministic order

        # 5. Calculate positions
        max_level = max(levels.values()) if levels else 0
        
        # Vertical spacing
        y_step = (self.height() / (max_level + 2) if max_level > 0 else self.height() / 2) * self.vertical_spacing_factor
        
        # Horizontal spacing and positioning
        for level in sorted(level_nodes.keys()):
            nodes_at_level = level_nodes[level]
            
            # Determine which margin to use for this level
            current_margin = self.node_margin
            # Check if this level contains nodes that were originally unvisited
            # This assumes 'start_level_for_unvisited' was set in the unvisited_nodes handling
            # and that levels for unvisited nodes are distinct and higher.
            # A more robust check might involve checking if all nodes in level_nodes[level] are in unvisited_nodes
            # For now, let's assume levels for unvisited nodes are higher than connected nodes.
            if 'start_level_for_unvisited' in locals() and level >= start_level_for_unvisited:
                current_margin = self.isolated_node_margin

            # Calculate total width required for nodes at this level, including margins
            total_level_width = 0
            for i, node_id in enumerate(nodes_at_level):
                node_width, _ = self.notes[node_id]['size']
                total_level_width += node_width
                if i < len(nodes_at_level) - 1:
                    total_level_width += current_margin # Margin between nodes

            # Start x position to center the level
            current_x = (self.width() - total_level_width) / 2
            if current_x < current_margin: # Ensure it doesn't go too far left
                current_x = current_margin

            for node_id in nodes_at_level:
                node_width, node_height = self.notes[node_id]['size']
                
                x = current_x + node_width / 2
                y = (level + 1) * y_step # +1 to avoid placing at very top edge
                self.notes[node_id]['pos'] = QPointF(x, y)
                
                current_x += node_width + current_margin # Move for next node, adding margin

        # log_debug(f"DEBUG: MindMapWidget._layout_nodes: Layered tree layout complete.")

    def paintEvent(self, event):
        # log_debug(f"DEBUG: MindMapWidget.paintEvent called. Widget size: {self.width()}x{self.height()}")
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self.font)
        # Correct order: scale then translate
        painter.scale(self.zoom_factor, self.zoom_factor)
        painter.translate(self.offset_x, self.offset_y)

        # Draw links first
        for source_id, target_id in self.links:
            if source_id in self.notes and target_id in self.notes:
                start_pos = self.notes[source_id]['pos']
                end_pos = self.notes[target_id]['pos']
                # log_debug(f"DEBUG: MindMapWidget.paintEvent: Drawing link from {source_id} to {target_id}")

                # Draw arrow
                painter.setPen(QPen(QColor(0, 0, 255), 2)) # Blue pen, increased thickness
                painter.drawLine(start_pos, end_pos)

                # Simple arrow head (optional, can be improved)
                # dx = end_pos.x() - start_pos.x()
                # dy = end_pos.y() - end_pos.y()
                # angle = math.atan2(dy, dx)
                # arrow_size = 8
                # painter.drawLine(end_pos, end_pos - QPointF(arrow_size * math.cos(angle - math.pi / 6), arrow_size * math.sin(angle - math.pi / 6)))
                # painter.drawLine(end_pos, end_pos - QPointF(arrow_size + math.cos(angle + math.pi / 6), arrow_size * math.sin(angle + math.pi / 6)))


        # Draw nodes
        for note_id, data in self.notes.items():
            pos = data['pos']
            title = data['title']
            # log_debug(f"DEBUG: MindMapWidget.paintEvent: Drawing node {note_id} with title '{title}' at {pos.x()}, {pos.y()}")

            node_width, node_height = data['size']

            # Adjust rect position to be centered around pos
            rect = QRectF(pos.x() - node_width / 2, pos.y() - node_height / 2, node_width, node_height)
            self.notes[note_id]['rect'] = rect # Store for click detection

            # Node background
            if note_id == self.current_note_id:
                painter.setBrush(QColor(0, 255, 0)) # Bright green for highlighted
            else:
                painter.setBrush(QColor(255, 0, 0)) # Bright red for default
            painter.setPen(QPen(QColor(0, 0, 0), 3))
            painter.drawRoundedRect(rect, 10, 10) # Rounded rectangle

            # Node text
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(rect, Qt.AlignCenter, title)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Transform mouse position from widget coordinates to logical coordinates
            # Reverse the transformations applied in paintEvent
            # 1. Translate by -offset_x, -offset_y
            # 2. Scale by 1 / zoom_factor
            transformed_x = (event.pos().x() / self.zoom_factor) - self.offset_x
            transformed_y = (event.pos().y() / self.zoom_factor) - self.offset_y
            transformed_pos = QPointF(transformed_x, transformed_y)

            for note_id, data in self.notes.items():
                if data['rect'].contains(transformed_pos):
                    self.current_note_id = note_id
                    self.note_selected.emit(note_id)
                    self.update()
                    break
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        zoom_in_factor = 1.1
        zoom_out_factor = 0.9

        old_zoom_factor = self.zoom_factor
        if event.angleDelta().y() > 0: # Scrolled up (zoom in)
            self.zoom_factor *= zoom_in_factor
        else: # Scrolled down (zoom out)
            self.zoom_factor *= zoom_out_factor

        # Clamp zoom factor
        self.zoom_factor = max(0.1, min(self.zoom_factor, 5.0))

        # Adjust position to zoom around mouse cursor
        # Get mouse position relative to the widget
        mouse_pos = event.pos()

        # Correct formula when translate is applied AFTER scale
        self.offset_x = (mouse_pos.x() / self.zoom_factor) - (mouse_pos.x() / old_zoom_factor) + self.offset_x
        self.offset_y = (mouse_pos.y() / self.zoom_factor) - (mouse_pos.y() / old_zoom_factor) + self.offset_y

        self.update()
        super().wheelEvent(event)

    def resizeEvent(self, event):
        # log_debug(f"DEBUG: MindMapWidget.resizeEvent called. New size: {event.size().width()}x{event.size().height()}")
        self.layout_timer.stop()
        self.layout_timer.start(self.layout_delay_ms)
        super().resizeEvent(event)

if __name__ == '__main__':
    # This is a placeholder for testing the widget independently.
    # In the actual application, db_manager will be passed from main.py.
    class MockDatabaseManager:
        def get_all_notes_metadata(self):
            return [
                ("1", "Note A", ""),
                ("2", "Note B", ""),
                ("3", "Note C", ""),
                ("4", "Note D", ""),
                ("5", "Note E", ""),
            ]
        def get_all_note_links(self):
            return [
                ("1", "2"),
                ("1", "3"),
                ("2", "4"),
                ("3", "5"),
                ("4", "5"),
            ]

    app = QApplication([])
    db_manager = MockDatabaseManager()
    widget = MindMapWidget(db_manager)

    all_notes_metadata, _ = db_manager.get_all_notes_metadata(), None # Mock doesn't return categories
    all_links = db_manager.get_all_note_links()
    widget.update_map(all_notes_metadata, all_links, current_note_id="1")

    widget.zoom_factor = 1.0
    widget.setMinimumSize(400, 300)
    widget.setMouseTracking(True) # Enable mouse tracking for hover effects

    widget.show()
    app.exec_()