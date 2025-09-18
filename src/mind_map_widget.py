from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QRect, QTimer
import math
from logger import log_debug

class MindMapWidget(QWidget):
    """Birbirine bağlı notların zihin haritasını görüntülemek için özel bir QWidget.
    
    Bu widget, notları düğüm olarak ve bağlantılarını yönlendirilmiş kenarlar olarak görselleştirir.
    Yakınlaştırma, kaydırma ve düğüm seçimini destekler, bir düğüme tıklandığında bir sinyal yayar.
    """
    note_selected = pyqtSignal(str) # Bir düğüme tıklandığında seçilen notun kimliğini yayar.

    def __init__(self, db_manager, parent=None):
        """MindMapWidget'ı başlatır.

        Args:
            db_manager: Not verileriyle etkileşim kurmak için DatabaseManager örneği.
            parent: Varsa, üst widget.
        """
        super().__init__(parent)
        self.db_manager = db_manager
        # Not verilerini saklar: {note_id: {'title': başlık, 'pos': QPointF, 'rect': QRectF, 'size': (genişlik, yükseklik)}}
        self.notes = {}  
        self.links = []  # Bağlantıları (kaynak_not_id, hedef_not_id) demetleri listesi olarak saklar.
        self.current_note_id = None # Şu anda seçili/odaklanmış notun kimliği.
        
        # --- Düğüm Stili ve Yerleşim Parametreleri ---
        self.node_radius = 10 # Düğümler için temel yarıçap, boyut hesaplamalarında kullanılır.
        self.node_padding = 5 # Düğüm metni ile düğüm kenarlığı arasındaki iç dolgu.
        self.node_margin = 40 # Bağlı düğümler arasındaki minimum dış boşluk.
        self.isolated_node_margin = 120 # Doğrudan bağlantısı olmayan düğümler için dış boşluk.
        self.vertical_spacing_factor = 2.5 # Seviyeler arasındaki dikey boşluk için çarpan.
        self.font = QFont("Arial", 10) # Not başlıklarını işlemek için kullanılan yazı tipi.
        
        # --- Görüntü Alanı Kontrol Parametreleri ---
        self.zoom_factor = 1.0 # Zihin haritasının mevcut yakınlaştırma seviyesi.
        self.offset_x = 0.0 # Görüntüyü kaydırmak için X-ofseti.
        self.offset_y = 0.0 # Görüntüyü kaydırmak için Y-ofseti.
        self._last_mouse_pos = None # Kaydırma sırasında son fare konumunu saklar.

        self.setMinimumSize(400, 300) # Widget'ın minimum boyutunu ayarlar.
        self.setMouseTracking(True) # Fare izlemeyi etkinleştirir (şu anda uygulanmamış olsa da).

        # --- Yerleşim Gecikmesi (Debouncing) ---
        # Yeniden boyutlandırma olaylarını geciktirmek için kullanılan bir QTimer,
        # aşırı yerleşim yeniden hesaplamalarını önler.
        self.layout_timer = QTimer(self)
        self.layout_timer.setSingleShot(True) # Zamanlayıcının her başlangıçta yalnızca bir kez tetiklenmesini sağlar.
        self.layout_timer.timeout.connect(self._perform_layout) # Zamanlayıcı zaman aşımını yerleşim metoduna bağlar.
        self.layout_delay_ms = 100 # Yeniden boyutlandırmada yerleşim yeniden hesaplamasından önceki gecikme (milisaniye).

        # --- Düğüm Seviye Renkleri ---
        # Grafik hiyerarşisindeki seviyelerine göre notları ayırt etmek için kullanılan renklerin bir listesi.
        # Grafik hiyerarşisindeki seviyelerine göre notları ayırt etmek için kullanılan renklerin bir listesi.
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
        """Zihin haritasını yeni not verileri ve bağlantılarla günceller.

        Bu yöntem, mevcut verileri temizler, yeni notları ve bağlantıları doldurur,
        ardından bir yerleşim yeniden hesaplamasını ve yeniden çizimi tetikler.

        Args:
            all_notes_metadata (list): Her bir not için (not_id, başlık, kategori_yolu) içeren demetlerin bir listesi.
            all_links (list): Bağlantıları temsil eden (kaynak_not_id, hedef_not_id) demetlerinin bir listesi.
            current_note_id (str, optional): Odaklanmış geçerli notun kimliği.
                                             Varsayılan olarak None.
        """
        # log_debug(f"DEBUG: MindMapWidget.update_map called. Notes metadata count: {len(all_notes_metadata)}, Links count: {len(all_links)}, Current note ID: {current_note_id}")
        self.notes.clear() # Mevcut notları temizle.
        self.links = all_links # Yeni bağlantıları ayarla.
        self.current_note_id = current_note_id # Geçerli odaklanmış not kimliğini ayarla.

        # Başlık ve konum ile dikdörtgen için yer tutucu dahil olmak üzere notlar sözlüğünü başlangıç verileriyle doldur.
        for note_id, title, _ in all_notes_metadata:
            self.notes[note_id] = {'title': title, 'pos': QPointF(), 'rect': QRectF()}

        self._layout_nodes() # Düğüm konumlarını yeniden hesapla.
        self.repaint() # Güncellenmiş haritayı göstermek için widget'ın yeniden çizimini zorla.

    def _perform_layout(self):
        """Düğüm yerleşim hesaplamasını tetikler ve widget'ı günceller.

        Bu yöntem, düğümlerin doğru şekilde konumlandırılmasını sağlamak için
        genellikle yeniden boyutlandırma olayından veya harita verileri değiştiğinde çağrılır.
        """
        self._layout_nodes() # Gerçek yerleşim hesaplamasını gerçekleştir.
        self.update() # Widget'ın yeniden çizimini iste.

    def _layout_nodes(self):
        """Zihin haritasındaki tüm notların (düğümlerin) konumlarını hesaplar ve ayarlar.

        Bu yöntem katmanlı bir grafik yerleşim algoritması uygular:
        1. Metin içeriğine göre düğüm boyutlarını önceden hesaplar.
        2. Verimli grafik geçişi için komşuluk listeleri oluşturur.
        3. Kök düğümleri (gelen bağlantısı olmayan düğümler veya geçerli not_id) tanımlar.
        4. BFS benzeri bir geçiş kullanarak düğümlere seviyeler atar, döngüleri ve izole düğümleri işler.
        5. Kenar geçişlerini en aza indirmek için her seviyedeki düğümleri sıralar (barycenter sezgisini kullanarak).
        6. Atanan seviyelerine ve sıralanmış düzenlerine göre her düğüm için son X ve Y konumlarını hesaplar.
        """
        # log_debug(f"DEBUG: MindMapWidget._layout_nodes called. Number of notes: {len(self.notes)}")
        if not self.notes:
            # log_debug("DEBUG: MindMapWidget._layout_nodes: No notes to layout.")
            return # Yerleştirilecek not yok, çık.

        # Başlangıç konumlandırması için widget'ın merkezini hesapla.
        center_x = self.width() / 2
        center_y = self.height() / 2

        num_nodes = len(self.notes)
        if num_nodes == 1:
            # Yalnızca bir not varsa, merkeze yerleştir.
            note_id = list(self.notes.keys())[0]
            data = self.notes[note_id]
            text_rect = self.fontMetrics().boundingRect(QRect(0, 0, 1000, 1000), Qt.AlignCenter, data['title'])
            node_width = max(self.node_radius * 2, text_rect.width() + self.node_padding * 2)
            node_height = max(self.node_radius * 2, text_rect.height() + self.node_padding * 2)
            self.notes[note_id]['size'] = (node_width, node_height)
            self.notes[note_id]['pos'] = QPointF(center_x, center_y)
            # log_debug(f"DEBUG: MindMapWidget._layout_nodes: Single note layout at {center_x}, {center_y}")
            return

        # 1. Düğüm boyutlarını önceden hesapla ve sakla.
        # Bu, yerleşimden önce düğüm boyutlarının bilinmesini sağlayarak uygun boşluk bırakmaya olanak tanır.
        for note_id, data in self.notes.items():
            # Notun başlık metni için sınırlayıcı dikdörtgeni hesapla.
            text_rect = self.fontMetrics().boundingRect(QRect(0, 0, 1000, 1000), Qt.AlignCenter, data['title'])
            # Düğüm genişliğini ve yüksekliğini belirle, node_radius ve metin boyutuna göre minimum bir boyut sağlar.
            node_width = max(self.node_radius * 2, text_rect.width() + self.node_padding * 2)
            node_height = max(self.node_radius * 2, text_rect.height() + self.node_padding * 2)
            self.notes[note_id]['size'] = (node_width, node_height) # Hesaplanan boyutu sakla.

        # 2. Grafik geçişi için komşuluk listesi oluştur.
        # 'adj' giden bağlantıları (düğüm -> komşular) saklar.
        # 'rev_adj' gelen bağlantıları (komşu -> düğümler) saklar, kökleri bulmak için kullanılır.
        adj = {note_id: [] for note_id in self.notes}
        rev_adj = {note_id: [] for note_id in self.notes} 
        for source_id, target_id in self.links:
            if source_id in adj and target_id in adj:
                adj[source_id].append(target_id)
                rev_adj[target_id].append(source_id)

        # 3. Kök düğümleri tanımla.
        # Kökler, gelen bağlantısı olmayan düğümler veya şu anda seçili olan nottur.
        roots = []
        if self.current_note_id and self.current_note_id in self.notes:
            roots.append(self.current_note_id) # Geçerli notu kök olarak önceliklendir.
        else:
            for note_id in self.notes:
                if not rev_adj[note_id]: # Düğümün gelen bağlantısı olup olmadığını kontrol et.
                    roots.append(note_id)
            if not roots and self.notes: # Gerçek kök yoksa (örn. döngüsel bir grafik), rastgele bir düğüm seç.
                roots.append(list(self.notes.keys())[0])
        
        # Döngüsel grafiklerde sonsuz döngüleri önlemek için BFS sırasında ziyaret edilen düğümleri takip etmek için bir küme kullan.
        visited = set()
        
        # 4. BFS/DFS kullanarak seviyeleri ve yatay sırayı ata.
        # 'levels' her not için atanan seviyeyi saklar.
        # 'level_nodes' notları atanan seviyelerine göre gruplandırır.
        self.levels = {} 
        level_nodes = {} 
        
        # Kök düğümleri seviye 0'a sahip kuyruğa ekle.
        queue = [(root_id, 0) for root_id in roots]
        for root_id in roots:
            self.levels[root_id] = 0
            if 0 not in level_nodes:
                level_nodes[0] = []
            level_nodes[0].append(root_id)
            visited.add(root_id)

        head = 0
        while head < len(queue):
            current_node_id, level = queue[head]
            head += 1

            # Geçerli düğümün komşularını (çocuklarını) işle.
            for neighbor_id in adj[current_node_id]:
                if neighbor_id not in visited: # Komşu henüz ziyaret edilmediyse, bir sonraki seviyeye ata.
                    self.levels[neighbor_id] = level + 1
                    if level + 1 not in level_nodes:
                        level_nodes[level + 1] = []
                    level_nodes[level + 1].append(neighbor_id)
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, level + 1))
                elif self.levels[neighbor_id] > level + 1: 
                    # Zaten ziyaret edilmiş bir düğüme daha kısa bir yol bulunursa, seviyesini güncelle
                    # ve yeni seviyesiyle çocuklarını yeniden işlemek için kuyruğa tekrar ekle.
                    self.levels[neighbor_id] = level + 1
                    queue.append((neighbor_id, level + 1))

        # Ziyaret edilmemiş düğümleri (izole düğümler veya ayrı bileşenler) işle.
        unvisited_nodes = [note_id for note_id in self.notes if note_id not in visited]
        if unvisited_nodes:
            # Ziyaret edilmemiş düğümleri yeni "sanal" seviyelere ata, genellikle ana grafiğin altına.
            # Bu, ana grafiğe bağlı olmasalar bile görüntülenmelerini sağlar.
            
            # Ziyaret edilmemiş düğümler için başlangıç seviyesini belirle, bağlı düğümlerden farklı olmasını sağla.
            start_level_for_unvisited = (max(self.levels.values()) + 2) if self.levels else 0
            
            # Belirleyici yerleşim için ziyaret edilmemiş düğümleri sırala.
            unvisited_nodes.sort()

            nodes_per_row = int(math.sqrt(len(unvisited_nodes))) + 1 # İzole düğümler için basit bir ızgara düzenlemesi.
            
            for i, note_id in enumerate(unvisited_nodes):
                row = i // nodes_per_row
                # Bu düğümler için ızgaradaki sıralarına göre sanal bir seviye oluştur.
                virtual_level = start_level_for_unvisited + row
                if virtual_level not in level_nodes:
                    level_nodes[virtual_level] = []
                level_nodes[virtual_level].append(note_id)
                self.levels[note_id] = virtual_level

        # Kenar geçişlerini azaltmak ve tutarlı yatay yerleşim için her seviyedeki düğümleri sırala.
        # Barycenter yöntemini kullanır: düğümler, ebeveynlerinin ortalama konumuna göre sıralanır.
        for level in sorted(level_nodes.keys()):
            nodes_at_level = level_nodes[level]
            if not nodes_at_level:
                continue

            barycenters = {}
            for node_id in nodes_at_level:
                parent_positions_sum = 0.0
                parent_count = 0
                for parent_id in rev_adj.get(node_id, []):
                    if parent_id in self.notes and 'pos' in self.notes[parent_id]:
                        parent_positions_sum += self.notes[parent_id]['pos'].x()
                        parent_count += 1
                
                if parent_count > 0:
                    barycenters[node_id] = parent_positions_sum / parent_count
                else:
                    # Ebeveyni olmayan düğümler (kökler veya izole) için bir kukla barycenter ata.
                    # Kararlılık için not_id'lerine göre sıralanacaklardır.
                    barycenters[node_id] = float('inf') 
            
            # Düğümleri barycenter'larına göre, ardından kararlılık için not_id'lerine göre sırala (belirleyici sıra).
            nodes_at_level.sort(key=lambda node_id: (barycenters.get(node_id, float('inf')), node_id))

        # 5. Her düğüm için son konumları (X ve Y koordinatları) hesapla.
        max_level = max(self.levels.values()) if self.levels else 0
        
        # Her seviyedeki maksimum düğüm yüksekliğine göre dinamik dikey boşluk hesapla.
        level_max_heights = {level: 0 for level in level_nodes}
        for level, nodes_at_level in level_nodes.items():
            for node_id in nodes_at_level:
                _, node_height = self.notes[node_id]['size']
                level_max_heights[level] = max(level_max_heights[level], node_height)

        y_positions = { -1: 0 } # Her seviyenin merkezi için hesaplanan y-koordinatını saklar.
        current_y = 0
        for level in sorted(level_nodes.keys()):
            # Önceki seviyenin maksimum yüksekliğinin yarısı, geçerli seviyenin maksimum yüksekliği ve sabit bir kenar boşluğuna göre boşluk ekle.
            current_y += level_max_heights[level] / 2 + self.node_margin 
            if level > 0 and (level - 1) in level_max_heights: 
                current_y += level_max_heights[level-1] / 2
            y_positions[level] = current_y
            current_y += level_max_heights[level] / 2 # Bir sonraki seviyenin hesaplaması için geçerli seviyenin düğümlerini geç.

        # Her seviye içindeki yatay boşluk ve konumlandırma.
        for level in sorted(level_nodes.keys()):
            nodes_at_level = level_nodes[level]
            
            # Hangi kenar boşluğunun kullanılacağını belirle (normal veya izole düğümler için).
            current_margin = self.node_margin
            # Bu kontrol, ziyaret edilmemiş düğümler için sanal seviyelerin daha yüksek olduğunu varsayar.
            if 'start_level_for_unvisited' in locals() and level >= start_level_for_unvisited:
                current_margin = self.isolated_node_margin

            # Bu seviyedeki düğümler için kenar boşlukları dahil olmak üzere gereken toplam genişliği hesapla.
            total_level_width = 0
            for i, node_id in enumerate(nodes_at_level):
                node_width, _ = self.notes[node_id]['size']
                total_level_width += node_width
                if i < len(nodes_at_level) - 1:
                    total_level_width += current_margin # Düğümler arasına kenar boşluğu ekle.

            # Seviyeyi widget içinde yatay olarak ortalamak için başlangıç x konumunu ayarla.
            current_x = (self.width() - total_level_width) / 2
            if current_x < current_margin: # Çok sola gitmemesini sağla.
                current_x = current_margin

            # Her düğüme son konumları ata.
            for node_id in nodes_at_level:
                node_width, node_height = self.notes[node_id]['size']
                
                x = current_x + node_width / 2 # Düğümü yatay olarak ortala.
                y = y_positions[level] # Seviye için dinamik olarak hesaplanan y konumunu kullan.
                self.notes[node_id]['pos'] = QPointF(x, y) # Hesaplanan merkez konumunu sakla.
                
                current_x += node_width + current_margin # Bir sonraki düğüm için x-imlecini hareket ettir, kenar boşluğu ekle.

        # log_debug(f"DEBUG: MindMapWidget._layout_nodes: Layered tree layout complete.")

    def paintEvent(self, event):
        """Widget üzerinde zihin haritasını çizer.

        Bu yöntem, widget'ın yeniden çizilmesi gerektiğinde çağrılır.
        Notlar arasındaki bağlantıları (okları) çizer ve ardından her not düğümünü
        başlığı ve uygun stil ile çizer.

        Args:
            event (QPaintEvent): Boyama olayı nesnesi.
        """
        # log_debug(f"DEBUG: MindMapWidget.paintEvent called. Widget size: {self.width()}x{self.height()}")
        painter = QPainter(self) # Çizim için bir QPainter nesnesi oluştur.
        painter.setRenderHint(QPainter.Antialiasing) # Daha pürüzsüz çizgiler ve metin için kenar yumuşatmayı etkinleştir.
        painter.setFont(self.font) # Metin işleme için yazı tipini ayarla.
        
        # Yakınlaştırma ve kaydırma için dönüşümleri uygula.
        # Sıra çok önemlidir: önce ölçekle, sonra çevir.
        painter.scale(self.zoom_factor, self.zoom_factor)
        painter.translate(self.offset_x, self.offset_y)

        # Bağlantı kalemi ve ok başı boyutu tanımla
        link_pen = QPen(QColor(150, 150, 150), 1) # Nötr gri kalem, 1 piksel kalınlık.
        arrow_size = 8 # Ok başının boyutu.

        # Bağlantıları düğümlerin arkasında görünmelerini sağlamak için önce çiz.
        for source_id, target_id in self.links:
            if source_id in self.notes and target_id in self.notes:
                start_pos = self.notes[source_id]['pos'] # Kaynak düğümün merkez konumunu al.
                end_pos = self.notes[target_id]['pos'] # Hedef düğümün merkez konumunu al.
                # log_debug(f"DEBUG: MindMapWidget.paintEvent: Drawing link from {source_id} to {target_id}")

                # Okun ana hattını çiz.
                painter.setPen(link_pen)
                painter.drawLine(start_pos, end_pos)

                # Hedef uçta ok başını çiz.
                dx = end_pos.x() - start_pos.x()
                dy = end_pos.y() - start_pos.y()
                angle = math.atan2(dy, dx) # Hattın açısını hesapla.
                
                # Ana hatta göre açılı iki ok başı çizgisi çiz.
                painter.drawLine(end_pos, end_pos - QPointF(arrow_size * math.cos(angle - math.pi / 6), arrow_size * math.sin(angle - math.pi / 6)))
                painter.drawLine(end_pos, end_pos - QPointF(arrow_size * math.cos(angle + math.pi / 6), arrow_size * math.sin(angle + math.pi / 6)))

        # Düğüm kenarlığı kalemi tanımla
        node_border_pen = QPen(QColor(0, 0, 0), 3) # Siyah kenarlık, 3 piksel kalınlık.

        # Düğümleri (notları) bağlantıların üzerine çiz.
        for note_id, data in self.notes.items():
            pos = data['pos'] # Düğümün merkez konumu.
            title = data['title'] # Notun başlık metni.
            # log_debug(f"DEBUG: MindMapWidget.paintEvent: Drawing node {note_id} with title '{title}' at {pos.x()}, {pos.y()}")

            node_width, node_height = data['size'] # Düğümün boyutları.

            # 'pos' etrafında ortalamak için dikdörtgenin sol üst köşesini hesapla.
            rect = QRectF(pos.x() - node_width / 2, pos.y() - node_height / 2, node_width, node_height)
            self.notes[note_id]['rect'] = rect # Tıklama algılama için sınırlayıcı dikdörtgeni sakla.

            # Grafikteki seviyesine göre düğüm rengini belirle.
            node_level = self.levels.get(note_id, 0) # Atanan seviyeyi al, bulunamazsa varsayılan olarak 0.
            node_color_index = node_level % len(self.level_colors) # Tanımlı renkler arasında döngü yap.
            node_color = self.level_colors[node_color_index]

            # Düğüm arka planı ve kenarlığı için fırça ve kalem ayarla.
            if note_id == self.current_note_id:
                painter.setBrush(node_color.lighter(150)) # Geçerli notu daha açık bir tonla vurgula.
            else:
                painter.setBrush(node_color) # Diğer notlar için standart rengi kullan.
            painter.setPen(node_border_pen)
            painter.drawRoundedRect(rect, 10, 10) # Düğüm şekli için yuvarlatılmış bir dikdörtgen çiz.

            # Düğüm metnini (başlık) çiz.
            painter.setPen(QColor(0, 0, 0)) # Metin için kalemi siyaha ayarla.
            painter.drawText(rect, Qt.AlignCenter, title) # Başlığı, düğüm dikdörtgeninin ortasına çiz.

    def mousePressEvent(self, event):
        """Düğüm seçimi ve kaydırma başlatma için fare basma olaylarını işler.

        Sol fare düğmesine basılırsa, bir not düğümüne tıklanıp tıklanmadığını kontrol eder
        ve `note_selected` sinyalini yayar. Sağ fare düğmesine basılırsa,
        kaydırmayı başlatmak için konumu kaydeder.

        Args:
            event (QMouseEvent): Fare olayı nesnesi.
        """
        if event.button() == Qt.LeftButton:
            # Fare konumunu widget koordinatlarından mantıksal (harita) koordinatlarına dönüştürür.
            # Bu, paintEvent'te uygulanan ölçeklendirme ve çeviriyi tersine çevirir.
            transformed_x = (event.pos().x() / self.zoom_factor) - self.offset_x
            transformed_y = (event.pos().y() / self.zoom_factor) - self.offset_y
            transformed_pos = QPointF(transformed_x, transformed_y)

            # Dönüştürülmüş fare konumunun herhangi bir notun sınırlayıcı dikdörtgeni içinde olup olmadığını kontrol et.
            for note_id, data in self.notes.items():
                if data['rect'].contains(transformed_pos):
                    self.current_note_id = note_id # Tıklanan notu geçerli olarak ayarla.
                    self.note_selected.emit(note_id) # Seçilen notun kimliğiyle sinyal yay.
                    self.update() # Vurguyu göstermek için yeniden boyama iste.
                    break # İlk isabetten sonra kontrol etmeyi durdur.
        elif event.button() == Qt.RightButton:
            self._last_mouse_pos = event.pos() # Kaydırma için son fare konumunu sakla.
        super().mousePressEvent(event) # Temel sınıf uygulamasını çağır.

    def mouseMoveEvent(self, event):
        """Kaydırma için fare hareket olaylarını işler.

        Sağ fare düğmesi basılı tutulursa ve bir sürükleme işlemi devam ediyorsa,
        görünümü etkili bir şekilde kaydırarak haritanın ofsetini fare hareketine göre günceller.

        Args:
            event (QMouseEvent): Fare olayı nesnesi.
        """
        # Sağ fare düğmesinin şu anda basılı olup olmadığını ve bir sürüklemenin başlatılıp başlatılmadığını kontrol et.
        if event.buttons() == Qt.RightButton and self._last_mouse_pos:
            delta = event.pos() - self._last_mouse_pos # Fare hareket farkını hesapla.
            # Ofsetleri güncelle, yakınlaştırma seviyesinden bağımsız olarak tutarlı kaydırma hızı sağlamak için
            # farkı yakınlaştırma faktörünün tersiyle ölçeklendir.
            self.offset_x += delta.x() / self.zoom_factor
            self.offset_y += delta.y() / self.zoom_factor
            self._last_mouse_pos = event.pos() # Bir sonraki hareket için son fare konumunu güncelle.
            self.update() # Kaydırılmış görünümü göstermek için yeniden boyama iste.
        super().mouseMoveEvent(event) # Temel sınıf uygulamasını çağır.

    def mouseReleaseEvent(self, event):
        """Fare bırakma olaylarını işler, öncelikli olarak kaydırmayı sonlandırmak için.

        Sağ fare düğmesi bırakılırsa, kaydedilen son fare konumunu temizler.

        Args:
            event (QMouseEvent): Fare olayı nesnesi.
        """
        if event.button() == Qt.RightButton:
            self._last_mouse_pos = None # Kaydırmayı durdurmak için son fare konumunu temizle.
        super().mouseReleaseEvent(event) # Temel sınıf uygulamasını çağır.

    def wheelEvent(self, event):
        """Yakınlaştırma ve uzaklaştırma için fare tekerleği olaylarını işler.

        Kaydırma yönüne göre zihin haritasını yakınlaştırır veya uzaklaştırır ve
        yakınlaştırmayı fare imlecinin etrafında ortalanmış tutmak için ofseti ayarlar.

        Args:
            event (QWheelEvent): Tekerlek olayı nesnesi.
        """
        zoom_in_factor = 1.1 # Yukarı kaydırmada yakınlaştırma faktörünü artır.
        zoom_out_factor = 0.9 # Aşağı kaydırmada yakınlaştırma faktörünü azalt.

        old_zoom_factor = self.zoom_factor # Hesaplama için mevcut yakınlaştırmayı sakla.
        if event.angleDelta().y() > 0: # Kaydırma yönünü kontrol et (yukarı kaydırma için pozitif).
            self.zoom_factor *= zoom_in_factor # Yakınlaştır.
        else: # Aşağı kaydırıldı.
            self.zoom_factor *= zoom_out_factor # Uzaklaştır.

        # Aşırı büyük veya küçük yakınlaştırma seviyelerini önlemek için yakınlaştırma faktörünü sınırla.
        self.zoom_factor = max(0.1, min(self.zoom_factor, 5.0))

        # Fare imleci etrafında yakınlaştırmak için konumu ayarla.
        # Bu formül, fare imlecinin altındaki noktanın yakınlaştırma sırasında sabit kalmasını sağlar.
        mouse_pos = event.pos() # Widget'a göre fare konumunu al.

        # paintEvent'te ölçeklendirme SONRA çeviri uygulandığında doğru formül.
        self.offset_x = (mouse_pos.x() / self.zoom_factor) - (mouse_pos.x() / old_zoom_factor) + self.offset_x
        self.offset_y = (mouse_pos.y() / self.zoom_factor) - (mouse_pos.y() / old_zoom_factor) + self.offset_y

        self.update() # Yakınlaştırılmış görünümü göstermek için yeniden boyama iste.
        super().wheelEvent(event) # Temel sınıf uygulamasını çağır.

    def resizeEvent(self, event):
        """Handles widget resize events.

        Triggers a debounced layout recalculation to adapt node positions to the new size.

        Args:
            event (QResizeEvent): The resize event object.
        """
        # log_debug(f"DEBUG: MindMapWidget.resizeEvent called. New size: {event.size().width()}x{event.size().height()}")
        self.layout_timer.stop() # Stop any pending layout timers.
        self.layout_timer.start(self.layout_delay_ms) # Start a new timer to debounce layout.
        super().resizeEvent(event) # Call the base class implementation.

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