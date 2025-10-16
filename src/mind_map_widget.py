from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QRect, QTimer
import math
from logger import log_debug
import pygraphviz as pgv

class MindMapWidget(QWidget):
    """A custom QWidget for displaying a mind map of interconnected notes.
    
    This widget visualizes notes as nodes and their links as directed edges.
    It supports zooming, panning, and node selection, emitting a signal when a node is clicked.
    """
    note_selected = pyqtSignal(str) # Emits the ID of the selected note when a node is clicked.

    def __init__(self, db_manager, parent=None):
        """Initializes the MindMapWidget.

        Args:
            db_manager: The DatabaseManager instance for interacting with note data.
            parent: The parent widget, if any.
        """
        super().__init__(parent)
        self.db_manager = db_manager
        # Stores note data: {note_id: {'title': title, 'pos': QPointF, 'rect': QRectF, 'size': (width, height)}}
        self.notes = {}  
        self.links = []  # Stores links as a list of (source_note_id, target_note_id) tuples.
        self.current_note_id = None # The ID of the currently selected/focused note.
        self.levels = {} # Stores the level of each note in the graph.
        
        # --- Node Styling and Layout Parameters ---
        self.node_radius = 10 # Base radius for nodes, used in size calculations.
        self.node_padding = 5 # Inner padding between node text and node border.
        self.node_margin = 80 # Minimum outer spacing between connected nodes.
        self.isolated_node_margin = 160 # Outer spacing for nodes with no direct connections.
        self.vertical_spacing_factor = 3.5 # Multiplier for vertical spacing between levels.
        self.font = QFont("Arial", 10) # The font used to render note titles.
        
        # --- Viewport Control Parameters ---
        self.zoom_factor = 1.0 # The current zoom level of the mind map.
        self.offset_x = 0.0 # The X-offset for panning the view.
        self.offset_y = 0.0 # The Y-offset for panning the view.
        self._last_mouse_pos = None # Stores the last mouse position during panning.

        self.setMinimumSize(400, 300) # Sets the minimum size of the widget.
        self.setMouseTracking(True) # Enables mouse tracking (though not currently implemented).

        # --- Layout Debouncing ---
        # A QTimer used to debounce resize events,
        # preventing excessive layout recalculations.
        self.layout_timer = QTimer(self)
        self.layout_timer.setSingleShot(True) # Ensures the timer only fires once per start.
        self.layout_timer.timeout.connect(self._perform_layout) # Connects the timer timeout to the layout method.
        self.layout_delay_ms = 100 # The delay (in milliseconds) before recalculating the layout on resize.

        # --- Node Level Colors ---
        # A list of colors used to differentiate notes based on their level in the graph hierarchy.
        self.level_colors = [
            QColor(255, 99, 71),   # Tomato
            QColor(60, 179, 113),  # MediumSeaGreen
            QColor(65, 105, 225),  # RoyalBlue
            QColor(255, 165, 0),   # Orange
            QColor(147, 112, 219), # MediumPurple
            QColor(0, 191, 255),   # DeepSkyBlue
            QColor(255, 20, 147),  # DeepPink
            QColor(0, 128, 128),   # Teal
            QColor(218, 165, 32),  # Goldenrod
            QColor(127, 255, 0)    # Chartreuse
        ]

    def update_map(self, all_notes_metadata, all_links, current_note_id=None):
        """Updates the mind map with new note data and links.

        This method clears the existing data, populates new notes and links,
        and then triggers a layout recalculation and repaint.

        Args:
            all_notes_metadata (list): A list of tuples, each containing (note_id, title, category_path) for a note.
            all_links (list): A list of (source_note_id, target_note_id) tuples representing links.
            current_note_id (str, optional): The ID of the currently focused note.
                                             Defaults to None.
        """
        self.notes.clear()
        self.links = all_links
        self.current_note_id = current_note_id

        for note_id, title, _ in all_notes_metadata:
            self.notes[note_id] = {'title': title, 'pos': QPointF(), 'rect': QRectF()}

        self._layout_nodes()
        self.center_on_nodes()
        self.repaint()

    def _perform_layout(self):
        """Triggers the node layout calculation and updates the widget.

        This method is typically called after a resize event or when the map data changes
        to ensure nodes are positioned correctly.
        """
        self._layout_nodes()
        self.update()

    def center_on_nodes(self):
        if not self.notes:
            return

        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')

        for note_id, data in self.notes.items():
            pos = data['pos']
            size = data['size']
            min_x = min(min_x, pos.x() - size[0] / 2)
            max_x = max(max_x, pos.x() + size[0] / 2)
            min_y = min(min_y, pos.y() - size[1] / 2)
            max_y = max(max_y, pos.y() + size[1] / 2)

        if min_x == float('inf'):
            return

        map_width = max_x - min_x
        map_height = max_y - min_y

        if map_width == 0 or map_height == 0:
            return

        x_scale = self.width() / map_width
        y_scale = self.height() / map_height
        self.zoom_factor = min(x_scale, y_scale) * 0.9

        self.offset_x = -min_x + (self.width() / self.zoom_factor - map_width) / 2
        self.offset_y = -min_y + (self.height() / self.zoom_factor - map_height) / 2
        self.update()

    def _layout_nodes(self):
        if not self.notes:
            return

        G = pgv.AGraph(directed=True, strict=True, splines='spline', overlap='scale', sep="+25,25")

        for note_id, data in self.notes.items():
            G.add_node(note_id, label=data['title'], shape='box')

        for source_id, target_id in self.links:
            if G.has_node(source_id) and G.has_node(target_id):
                G.add_edge(source_id, target_id)

        G.layout(prog='neato')

        for node in G.nodes():
            try:
                pos = node.attr['pos'].split(',')
                x = float(pos[0])
                y = float(pos[1])
                self.notes[node]['pos'] = QPointF(x, y)
                size = self.fontMetrics().boundingRect(QRect(0, 0, 1000, 1000), Qt.AlignCenter, node.attr['label'])
                self.notes[node]['size'] = (size.width() + self.node_padding * 2, size.height() + self.node_padding * 2)
            except (KeyError, IndexError, ValueError) as e:
                print(f"Error processing node {node}: {e}")

    def paintEvent(self, event):
        """Draws the mind map on the widget.

        This method is called whenever the widget needs to be repainted.
        It draws the links (arrows) between notes, and then draws each note node
        with its title and appropriate styling.

        Args:
            event (QPaintEvent): The paint event object.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self.font)
        
        painter.scale(self.zoom_factor, self.zoom_factor)
        painter.translate(self.offset_x, self.offset_y)

        link_pen = QPen(QColor(150, 150, 150), 1)
        arrow_size = 8

        for source_id, target_id in self.links:
            if source_id in self.notes and target_id in self.notes:
                start_pos = self.notes[source_id]['pos']
                end_pos = self.notes[target_id]['pos']

                painter.setPen(link_pen)
                painter.drawLine(start_pos, end_pos)

                dx = end_pos.x() - start_pos.x()
                dy = end_pos.y() - start_pos.y()
                angle = math.atan2(dy, dx)
                
                painter.drawLine(end_pos, end_pos - QPointF(arrow_size * math.cos(angle - math.pi / 6), arrow_size * math.sin(angle - math.pi / 6)))
                painter.drawLine(end_pos, end_pos - QPointF(arrow_size * math.cos(angle + math.pi / 6), arrow_size * math.sin(angle + math.pi / 6)))

        node_border_pen = QPen(QColor(0, 0, 0), 3)

        for note_id, data in self.notes.items():
            pos = data['pos']
            title = data['title']

            node_width, node_height = data.get('size', (100, 50))

            rect = QRectF(pos.x() - node_width / 2, pos.y() - node_height / 2, node_width, node_height)
            self.notes[note_id]['rect'] = rect

            node_level = self.levels.get(note_id, 0)
            node_color_index = node_level % len(self.level_colors)
            node_color = self.level_colors[node_color_index]

            if note_id == self.current_note_id:
                painter.setBrush(node_color.lighter(150))
            else:
                painter.setBrush(node_color)
            painter.setPen(node_border_pen)
            painter.drawRoundedRect(rect, 10, 10)

            painter.setPen(QColor(0, 0, 0))
            painter.drawText(rect, Qt.AlignCenter, title)

    def mousePressEvent(self, event):
        """Handles mouse press events for node selection and initiating panning.

        If the left mouse button is pressed, it checks if a note node was clicked
        and emits the `note_selected` signal. If the right mouse button is pressed, it
        saves the position to initiate panning.

        Args:
            event (QMouseEvent): The mouse event object.
        """
        if event.button() == Qt.LeftButton:
            transformed_x = (event.pos().x() / self.zoom_factor) - self.offset_x
            transformed_y = (event.pos().y() / self.zoom_factor) - self.offset_y
            transformed_pos = QPointF(transformed_x, transformed_y)

            for note_id, data in self.notes.items():
                if data['rect'].contains(transformed_pos):
                    self.current_note_id = note_id
                    self.note_selected.emit(note_id)
                    self.update()
                    break
        elif event.button() == Qt.RightButton:
            self._last_mouse_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handles mouse move events for panning.

        If the right mouse button is held down and a drag is in progress, it updates
        the map's offset based on the mouse movement, effectively panning the view.

        Args:
            event (QMouseEvent): The mouse event object.
        """
        if event.buttons() == Qt.RightButton and self._last_mouse_pos:
            delta = event.pos() - self._last_mouse_pos
            self.offset_x += delta.x() / self.zoom_factor
            self.offset_y += delta.y() / self.zoom_factor
            self._last_mouse_pos = event.pos()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handles mouse release events, primarily to terminate panning.

        If the right mouse button is released, it clears the saved last mouse position.

        Args:
            event (QMouseEvent): The mouse event object.
        """
        if event.button() == Qt.RightButton:
            self._last_mouse_pos = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """Handles mouse wheel events for zooming in and out.

        Zooms the mind map in or out based on the scroll direction and adjusts the offset
        to keep the zoom centered around the mouse cursor.

        Args:
            event (QWheelEvent): The wheel event object.
        """
        zoom_in_factor = 1.1
        zoom_out_factor = 0.9

        old_zoom_factor = self.zoom_factor
        if event.angleDelta().y() > 0:
            self.zoom_factor *= zoom_in_factor
        else:
            self.zoom_factor *= zoom_out_factor

        self.zoom_factor = max(0.1, min(self.zoom_factor, 5.0))

        mouse_pos = event.pos()

        self.offset_x = (mouse_pos.x() / self.zoom_factor) - (mouse_pos.x() / old_zoom_factor) + self.offset_x
        self.offset_y = (mouse_pos.y() / self.zoom_factor) - (mouse_pos.y() / old_zoom_factor) + self.offset_y

        self.update()
        super().wheelEvent(event)

    def resizeEvent(self, event):
        """Handles widget resize events.

        Triggers a debounced layout recalculation to adapt node positions to the new size.

        Args:
            event (QResizeEvent): The resize event object.
        """
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

    all_notes_metadata, _ = db_manager.get_all_notes_metadata(), None
    all_links = db_manager.get_all_note_links()
    widget.update_map(all_notes_metadata, all_links, current_note_id="1")

    widget.zoom_factor = 1.0
    widget.setMinimumSize(400, 300)
    widget.setMouseTracking(True)

    widget.show()
    app.exec_()
