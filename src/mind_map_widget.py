# pyrefly: ignore [missing-import]
import flet as ft
import flet.canvas as cv
import math
from logger import log_debug

# The MindMapWidget is a custom Flet control derived from ft.GestureDetector.
# It uses Flet Canvas shapes to draw a mind map of Zettelkasten notes and connections,
# and detects node clicks using coordinate bounding boxes.
class MindMapWidget(ft.Container):
    # The __init__ method initializes the widget with a database manager and a selection callback.
    def __init__(self, db_manager, on_note_selected):
        super().__init__()
        self.db_manager = db_manager
        self.on_note_selected = on_note_selected
        self.notes = {}
        self.links = []
        self.current_note_id = None
        self.node_positions = {} # Persisted coordinates {note_id: {'pos': ..., 'rect': ..., 'size': ...}}
        self._last_nodes_set = None
        self._last_links_set = None
        
        # Dimensions of the widget view
        self.width = 350
        self.height = 350
        self.clip_behavior = ft.ClipBehavior.HARD_EDGE
        self.bgcolor = ft.Colors.TRANSPARENT
        
        # Inner Canvas where shapes are drawn
        self.canvas = cv.Canvas(shapes=[])
        
        # GestureDetector catches taps for the nodes
        self.gd = ft.GestureDetector(
            content=self.canvas,
            on_tap_down=self.handle_tap_down
        )
        
        # InteractiveViewer handles panning and zooming
        self.viewer = ft.InteractiveViewer(
            min_scale=0.1,
            max_scale=4.0,
            boundary_margin=ft.Margin(300, 300, 300, 300),
            content=self.gd,
            pan_enabled=True,
            scale_enabled=True,
            constrained=False
        )
        
        self.content = self.viewer
        
    # The update_map method rebuilds the layout and draws the map.
    # all_notes_metadata: A list of tuples containing (note_id, title, category)
    # all_links: A list of tuples containing (source_note_id, target_note_id)
    # current_note_id: The ID of the currently selected note.
    def update_map(self, all_notes_metadata, all_links, current_note_id=None):
        self.notes.clear()
        self.links = all_links
        self.current_note_id = current_note_id
        
        # Initialize note metadata
        for note_id, title, _ in all_notes_metadata:
            self.notes[note_id] = {'title': title, 'pos': (175, 175), 'rect': (0, 0, 0, 0)}
            
        self._layout_nodes()
        self._draw_map()
        
    # The _layout_nodes method computes node coordinates using PyGraphviz layout.
    def _layout_nodes(self):
        if not self.notes:
            return

        current_nodes_set = set(self.notes.keys())
        current_links_set = set(self.links)

        # Check if node topology or links changed
        if self._last_nodes_set is not None and self._last_links_set is not None:
            if self._last_nodes_set == current_nodes_set and self._last_links_set == current_links_set:
                if self.node_positions:
                    for note_id in self.notes:
                        if note_id in self.node_positions:
                            self.notes[note_id]['pos'] = self.node_positions[note_id]['pos']
                            self.notes[note_id]['rect'] = self.node_positions[note_id]['rect']
                            self.notes[note_id]['size'] = self.node_positions[note_id]['size']
                    return

        self._last_nodes_set = current_nodes_set
        self._last_links_set = current_links_set

        # 1. Pre-calculate node dimensions and store them
        for note_id, data in self.notes.items():
            title = data['title']
            node_width = max(110, len(title) * 8 + 30)
            node_height = 40
            self.notes[note_id]['size'] = (node_width, node_height)

        raw_positions = {}

        try:
            import pygraphviz as pgv
            # Configure layout spacing and direction (Left to Right)
            G = pgv.AGraph(directed=True, strict=True, rankdir='LR', nodesep='0.5', ranksep='1.0')
            
            for note_id, data in self.notes.items():
                node_width, node_height = data['size']
                # Convert to inches (1 inch = 72 points)
                w_inches = (node_width + 20) / 72.0
                h_inches = (node_height + 10) / 72.0
                # Use fixedsize='shape' to prevent Graphviz 'size too small for label' warnings
                G.add_node(note_id, label=data['title'], shape='box', width=str(w_inches), height=str(h_inches), fixedsize='shape')
            
            for source_id, target_id in self.links:
                if G.has_node(source_id) and G.has_node(target_id):
                    G.add_edge(source_id, target_id)
            
            G.layout(prog='dot')
            
            for node in G.nodes():
                try:
                    pos = node.attr['pos'].split(',')
                    x = float(pos[0])
                    y = float(pos[1]) # Keep y positive, coordinates will be centered and scaled later
                    raw_positions[str(node)] = (x, y)
                except (KeyError, IndexError, ValueError):
                    pass
        except Exception as e:
            log_debug(f"PyGraphviz layout failed: {e}, using simple grid layout.")
            # Fallback simple layout if pgv fails
            nodes_list = list(self.notes.keys())
            nodes_per_row = int(len(nodes_list)**0.5) + 1
            for i, node_id in enumerate(nodes_list):
                row = i // nodes_per_row
                col = i % nodes_per_row
                raw_positions[node_id] = (col * 150, row * 100)

        # 6. Find raw bounding box and shift coordinates to fit inside InteractiveViewer
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')

        for note_id, (x, y) in raw_positions.items():
            w, h = self.notes[note_id]['size']
            min_x = min(min_x, x - w/2)
            max_x = max(max_x, x + w/2)
            min_y = min(min_y, y - h/2)
            max_y = max(max_y, y + h/2)

        if min_x == float('inf'):
            min_x, max_x, min_y, max_y = 0, 0, 0, 0

        padding = 50
        total_width = max(350, max_x - min_x + 2 * padding)
        total_height = max(350, max_y - min_y + 2 * padding)

        self.gd.width = total_width
        self.gd.height = total_height
        self.canvas.width = total_width
        self.canvas.height = total_height

        self.node_positions.clear()

        for note_id, (raw_x, raw_y) in raw_positions.items():
            mapped_x = (raw_x - min_x) + padding
            mapped_y = (raw_y - min_y) + padding

            w, h = self.notes[note_id]['size']
            
            rx = mapped_x - w / 2
            ry = mapped_y - h / 2

            self.notes[note_id]['pos'] = (mapped_x, mapped_y)
            self.notes[note_id]['rect'] = (rx, ry, rx + w, ry + h)

            # Persist positions for future layout updates
            self.node_positions[note_id] = {
                'pos': (mapped_x, mapped_y),
                'rect': (rx, ry, rx + w, ry + h),
                'size': (w, h)
            }
                
    # The _draw_map method populates the Flet Canvas shapes list and renders it.
    def _draw_map(self):
        shapes = []
        
        # 1. Draw edge links (lines with arrowheads)
        edge_width = 1.5
        edge_paint = ft.Paint(color=ft.Colors.OUTLINE, stroke_width=edge_width, style=ft.PaintingStyle.STROKE)
        
        for source_id, target_id in self.links:
            if source_id in self.notes and target_id in self.notes:
                start_x, start_y = self.notes[source_id]['pos']
                end_x, end_y = self.notes[target_id]['pos']
                
                # Calculate direct distance
                dx = end_x - start_x
                dy = end_y - start_y
                dist = math.hypot(dx, dy)
                if dist == 0:
                    continue
                    
                target_w, target_h = self.notes[target_id]['size']
                # Stop edge line at target node border
                radius = min(target_w, target_h) / 2 + 5
                
                adj_end_x = end_x - (dx / dist) * radius
                adj_end_y = end_y - (dy / dist) * radius
                
                # Edge line
                shapes.append(cv.Line(x1=start_x, y1=start_y, x2=adj_end_x, y2=adj_end_y, paint=edge_paint))
                
                # Arrowhead lines
                angle = math.atan2(dy, dx)
                arrow_size = 8
                
                x1 = adj_end_x - arrow_size * math.cos(angle - math.pi / 6)
                y1 = adj_end_y - arrow_size * math.sin(angle - math.pi / 6)
                x2 = adj_end_x - arrow_size * math.cos(angle + math.pi / 6)
                y2 = adj_end_y - arrow_size * math.sin(angle + math.pi / 6)
                
                shapes.append(cv.Line(x1=adj_end_x, y1=adj_end_y, x2=x1, y2=y1, paint=edge_paint))
                shapes.append(cv.Line(x1=adj_end_x, y1=adj_end_y, x2=x2, y2=y2, paint=edge_paint))
                
        # 2. Draw nodes
        border_width = 2.0
        border_paint = ft.Paint(color=ft.Colors.OUTLINE, stroke_width=border_width, style=ft.PaintingStyle.STROKE)
        
        for note_id, data in self.notes.items():
            mapped_x, mapped_y = data['pos']
            rx, ry, rx2, ry2 = data['rect']
            w, h = data['size']
            title = data['title']
            
            # Use specific color based on selection status
            is_current = (note_id == self.current_note_id)
            color = ft.Colors.PRIMARY if is_current else ft.Colors.SURFACE_CONTAINER_HIGHEST
            
            fill_paint = ft.Paint(color=color, style=ft.PaintingStyle.FILL)
            
            # Dynamic border radius
            node_border_radius = 8
            
            # Node background rectangle
            shapes.append(cv.Rect(x=rx, y=ry, width=w, height=h, border_radius=node_border_radius, paint=fill_paint))
            # Node border rectangle
            shapes.append(cv.Rect(x=rx, y=ry, width=w, height=h, border_radius=node_border_radius, paint=border_paint))
            
            # Node text label with dynamic font size (minimum 9px)
            font_size = 13
            shapes.append(cv.Text(
                x=mapped_x,
                y=mapped_y,
                value=title,
                alignment=ft.Alignment.CENTER,
                style=ft.TextStyle(size=font_size, color=ft.Colors.ON_PRIMARY if is_current else ft.Colors.ON_SURFACE_VARIANT, weight=ft.FontWeight.BOLD)
            ))
            
        self.canvas.shapes = shapes
        self.canvas.update()
        
    # The handle_tap_down method checks if any node is clicked in the canvas space.
    def handle_tap_down(self, e: ft.TapEvent):
        try:
            click_x = e.local_position.x
            click_y = e.local_position.y
        except AttributeError:
            # Fallback for older Flet versions
            click_x = getattr(e, 'local_x', 0)
            click_y = getattr(e, 'local_y', 0)
        
        for note_id, data in self.notes.items():
            rx, ry, rx2, ry2 = data['rect']
            if rx <= click_x <= rx2 and ry <= click_y <= ry2:
                self.current_note_id = note_id
                self.on_note_selected(note_id)
                self._draw_map()
                break
