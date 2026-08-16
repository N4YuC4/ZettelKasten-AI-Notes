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
        self._last_nodes_fingerprint = None
        
        self.expand = True
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
            constrained=False,
            expand=True
        )
        
        self.content = self.viewer

    # Explicit cache invalidation method
    def invalidate_cache(self):
        """Invalidates cached node positions, bounding boxes, and layout fingerprints."""
        self.node_positions.clear()
        self._last_nodes_set = None
        self._last_links_set = None
        self._last_nodes_fingerprint = None
        
    # The update_map method rebuilds the layout and draws the map.
    # all_notes_metadata: A list of tuples containing (note_id, title, category)
    # all_links: A list of tuples containing (source_note_id, target_note_id)
    # current_note_id: The ID of the currently selected note.
    def update_map(self, all_notes_metadata, all_links, current_note_id=None):
        self.notes.clear()
        self.links = list(all_links) if all_links is not None else []
        self.current_note_id = current_note_id
        
        # Initialize note metadata
        for note_id, title, _ in all_notes_metadata:
            self.notes[note_id] = {'title': title, 'pos': (175, 175), 'rect': (0, 0, 0, 0)}
            
        self._layout_nodes()
        self._draw_map()
        
    # The _layout_nodes method computes node coordinates using PyGraphviz layout or fallback grid.
    def _layout_nodes(self):
        if not self.notes:
            self.canvas.shapes = []
            try:
                self.canvas.update()
            except Exception:
                pass
            return

        current_nodes_fingerprint = {(nid, data['title']) for nid, data in self.notes.items()}
        current_links_set = set(self.links)

        # Check if node topology, titles, or links changed
        if self._last_nodes_fingerprint is not None and self._last_links_set is not None:
            if self._last_nodes_fingerprint == current_nodes_fingerprint and self._last_links_set == current_links_set:
                if self.node_positions and all(nid in self.node_positions for nid in self.notes):
                    for note_id in self.notes:
                        self.notes[note_id]['pos'] = self.node_positions[note_id]['pos']
                        self.notes[note_id]['rect'] = self.node_positions[note_id]['rect']
                        self.notes[note_id]['size'] = self.node_positions[note_id]['size']
                    return

        self._last_nodes_fingerprint = current_nodes_fingerprint
        self._last_nodes_set = set(self.notes.keys())
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
            G = pgv.AGraph(directed=True, strict=True, rankdir='LR', nodesep='0.6', ranksep='1.2')
            
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
            raw_positions.clear()

        # Position any unpositioned/disconnected nodes with collision-free spacing
        missing_nodes = [nid for nid in self.notes.keys() if nid not in raw_positions]
        if missing_nodes:
            max_node_width = max((self.notes[nid]['size'][0] for nid in self.notes), default=110)
            max_node_height = max((self.notes[nid]['size'][1] for nid in self.notes), default=40)
            col_step = max_node_width + 50
            row_step = max_node_height + 45
            
            # Find connected components within missing nodes
            adj = {nid: set() for nid in missing_nodes}
            for s, t in self.links:
                if s in adj and t in adj:
                    adj[s].add(t)
                    adj[t].add(s)
            
            visited = set()
            components = []
            for nid in missing_nodes:
                if nid not in visited:
                    comp = []
                    queue = [nid]
                    visited.add(nid)
                    while queue:
                        curr = queue.pop(0)
                        comp.append(curr)
                        for neighbor in adj[curr]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)
                    components.append(comp)

            base_y = max((y for _, y in raw_positions.values()), default=0.0) + row_step if raw_positions else 0.0
            current_y = base_y
            
            # Lay out connected clusters first
            for comp in [c for c in components if len(c) > 1]:
                comp_cols = max(1, int(math.ceil(math.sqrt(len(comp)))))
                for i, node_id in enumerate(comp):
                    row = i // comp_cols
                    col = i % comp_cols
                    raw_positions[node_id] = (col * col_step, current_y + row * row_step)
                comp_rows = (len(comp) + comp_cols - 1) // comp_cols
                current_y += (comp_rows + 0.5) * row_step
                
            # Lay out isolated singletons in a grid
            singletons = [c[0] for c in components if len(c) == 1]
            if singletons:
                sing_cols = max(1, int(math.ceil(math.sqrt(len(singletons)))))
                for i, node_id in enumerate(singletons):
                    row = i // sing_cols
                    col = i % sing_cols
                    raw_positions[node_id] = (col * col_step, current_y + row * row_step)

        # 6. Find raw bounding box and shift coordinates to fit inside InteractiveViewer
        min_x = min((raw_positions[nid][0] - self.notes[nid]['size'][0] / 2 for nid in self.notes if nid in raw_positions), default=0.0)
        max_x = max((raw_positions[nid][0] + self.notes[nid]['size'][0] / 2 for nid in self.notes if nid in raw_positions), default=0.0)
        min_y = min((raw_positions[nid][1] - self.notes[nid]['size'][1] / 2 for nid in self.notes if nid in raw_positions), default=0.0)
        max_y = max((raw_positions[nid][1] + self.notes[nid]['size'][1] / 2 for nid in self.notes if nid in raw_positions), default=0.0)

        padding = 50
        total_width = max(350, max_x - min_x + 2 * padding)
        total_height = max(350, max_y - min_y + 2 * padding)

        self.gd.width = total_width
        self.gd.height = total_height
        self.canvas.width = total_width
        self.canvas.height = total_height

        self.node_positions.clear()

        for note_id, (raw_x, raw_y) in raw_positions.items():
            if note_id not in self.notes:
                continue
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
                # Stop edge line at target node border, clamped to prevent inverted arrowheads
                radius = min(target_w, target_h) / 2 + 5
                radius = min(radius, max(0.0, dist - 5))
                
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
        try:
            self.canvas.update()
        except Exception:
            pass
        
    # The handle_tap_down method checks if any node is clicked in the canvas space.
    def handle_tap_down(self, e: ft.TapEvent):
        try:
            if hasattr(e, 'local_position') and e.local_position is not None:
                click_x = float(e.local_position.x)
                click_y = float(e.local_position.y)
            elif hasattr(e, 'local_x') and e.local_x is not None:
                click_x = float(e.local_x)
                click_y = float(getattr(e, 'local_y', 0.0))
            elif hasattr(e, 'global_position') and e.global_position is not None:
                click_x = float(e.global_position.x)
                click_y = float(e.global_position.y)
            else:
                click_x = float(getattr(e, 'local_x', 0.0))
                click_y = float(getattr(e, 'local_y', 0.0))
        except Exception as ex:
            log_debug(f"Tap event position extraction fallback: {ex}")
            click_x = float(getattr(e, 'local_x', 0.0))
            click_y = float(getattr(e, 'local_y', 0.0))
        
        for note_id, data in self.notes.items():
            rect = data.get('rect')
            if not rect:
                continue
            rx, ry, rx2, ry2 = rect
            if rx <= click_x <= rx2 and ry <= click_y <= ry2:
                if self.on_note_selected and callable(self.on_note_selected):
                    self.on_note_selected(note_id)
                break
