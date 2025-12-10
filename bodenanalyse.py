#!/usr/bin/env python3
"""
Bodenanalyse - Ausgleichsplan Generator
========================================

Analysiert 3D-Aufmaßdaten und erstellt Ausgleichspläne für schwimmend verlegte Böden.

Verwendung:
    python bodenanalyse.py

Konfiguration: Passe die Variablen im Abschnitt "KONFIGURATION" an.

Benötigte Pakete:
    pip install ezdxf matplotlib numpy scipy reportlab
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

# =============================================================================
# KONFIGURATION - Hier anpassen!
# =============================================================================

# Pfad zur DXF-Eingabedatei
INPUT_DXF = "Boden-links.dxf"

# Ausgabeverzeichnis (wird erstellt falls nicht vorhanden)
OUTPUT_DIR = "ausgabe"

# Schwellwert in mm (ab welcher Abweichung wird markiert?)
SCHWELLE = 2.0  # Empfehlung: 1.0 = sehr genau, 2.0 = Standard, 3.0 = tolerant

# Suchradius für lokale Referenz in mm
RADIUS = 400  # Empfehlung: 400mm bei 10cm Raster

# Referenzwand und Nullpunkt-Ecke festlegen
REFERENCE_WALL = "S"  # N = Nord (höchstes Y), S = Süd (niedrigstes Y), E = Ost (höchstes X), W = West (niedrigstes X)
ORIGIN_CORNER = "SW"  # NE = Nordost, NW = Nordwest, SE = Südost, SW = Südwest

# Ausgabeformate aktivieren/deaktivieren
EXPORT_PNG = True
EXPORT_PDF = True
EXPORT_DXF = True

# =============================================================================
# AB HIER NICHTS ÄNDERN (außer du weißt was du tust)
# =============================================================================

def check_dependencies():
    """Prüft ob alle benötigten Pakete installiert sind"""
    missing = []
    
    try:
        import ezdxf
    except ImportError:
        missing.append("ezdxf")
    
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import scipy
    except ImportError:
        missing.append("scipy")
    
    if EXPORT_PDF:
        try:
            import reportlab
        except ImportError:
            missing.append("reportlab")
    
    if missing:
        print("FEHLER: Fehlende Pakete!")
        print(f"Bitte installieren mit: pip install {' '.join(missing)}")
        sys.exit(1)


def load_dxf(filepath):
    """Lädt DXF und extrahiert Punkte und Grundriss"""
    import ezdxf
    
    if not os.path.exists(filepath):
        print(f"FEHLER: Datei nicht gefunden: {filepath}")
        sys.exit(1)
    
    print(f"Lade: {filepath}")
    doc = ezdxf.readfile(filepath)
    msp = doc.modelspace()
    
    # Punkte extrahieren
    points = []
    for entity in msp.query('POINT'):
        points.append([
            entity.dxf.location.x,
            entity.dxf.location.y,
            entity.dxf.location.z
        ])
    
    if not points:
        print("FEHLER: Keine POINT-Entities in der DXF gefunden!")
        sys.exit(1)
    
    points = np.array(points)
    print(f"  → {len(points)} Messpunkte gefunden")
    
    # Grundriss (Linien) extrahieren
    lines = []
    
    # LINE Entities
    for entity in msp.query('LINE'):
        start = (entity.dxf.start.x, entity.dxf.start.y)
        end = (entity.dxf.end.x, entity.dxf.end.y)
        lines.append((start, end))
    
    # LWPOLYLINE Entities (Lightweight Polylines)
    for entity in msp.query('LWPOLYLINE'):
        vertices = list(entity.vertices())
        for i in range(len(vertices) - 1):
            start = (float(vertices[i][0]), float(vertices[i][1]))
            end = (float(vertices[i+1][0]), float(vertices[i+1][1]))
            lines.append((start, end))
        # Wenn geschlossen, verbinde letzten mit erstem Punkt
        if entity.closed and len(vertices) > 2:
            start = (float(vertices[-1][0]), float(vertices[-1][1]))
            end = (float(vertices[0][0]), float(vertices[0][1]))
            lines.append((start, end))
    
    # POLYLINE Entities
    for entity in msp.query('POLYLINE'):
        vertices = list(entity.vertices())
        for i in range(len(vertices) - 1):
            start = (vertices[i].dxf.location.x, vertices[i].dxf.location.y)
            end = (vertices[i+1].dxf.location.x, vertices[i+1].dxf.location.y)
            lines.append((start, end))
        # Wenn geschlossen, verbinde letzten mit erstem Punkt
        if entity.is_closed and len(vertices) > 2:
            start = (vertices[-1].dxf.location.x, vertices[-1].dxf.location.y)
            end = (vertices[0].dxf.location.x, vertices[0].dxf.location.y)
            lines.append((start, end))
    
    print(f"  → {len(lines)} Grundrisslinien gefunden")
    
    return points, lines


def find_north_wall(lines):
    """
    Findet die Nordwand (Linie mit höchstem Y) für die Referenz-Beschriftung.
    Verändert KEINE Koordinaten, gibt nur Referenzinformationen zurück.
    """
    if len(lines) == 0:
        return None
    
    # Finde Referenzwand basierend auf REFERENCE_WALL Parameter
    wall_name_map = {
        'N': 'Nord (höchstes Y)',
        'S': 'Süd (niedrigstes Y)',
        'E': 'Ost (höchstes X)',
        'W': 'West (niedrigstes X)'
    }
    
    if REFERENCE_WALL == 'N':
        # Nordwand: höchstes durchschnittliches Y
        target_val = max(lines, key=lambda line: (line[0][1] + line[1][1]) / 2)
        ref_wall = target_val
    elif REFERENCE_WALL == 'S':
        # Südwand: niedrigstes durchschnittliches Y
        target_val = min(lines, key=lambda line: (line[0][1] + line[1][1]) / 2)
        ref_wall = target_val
    elif REFERENCE_WALL == 'E':
        # Ostwand: höchstes durchschnittliches X
        target_val = max(lines, key=lambda line: (line[0][0] + line[1][0]) / 2)
        ref_wall = target_val
    elif REFERENCE_WALL == 'W':
        # Westwand: niedrigstes durchschnittliches X
        target_val = min(lines, key=lambda line: (line[0][0] + line[1][0]) / 2)
        ref_wall = target_val
    else:
        print(f"  ! Ungültige REFERENCE_WALL: {REFERENCE_WALL}. Verwende 'N'")
        target_val = max(lines, key=lambda line: (line[0][1] + line[1][1]) / 2)
        ref_wall = target_val
    
    p1, p2 = ref_wall
    
    # Nullpunkt soll auf der Referenzwand liegen, an der gewünschten Ecke
    # Je nach Referenzwand und Ecke bestimmen wir den Punkt auf der Linie
    
    # Berechne Bounds für alle Punkte
    all_x = [p[0] for line in lines for p in line]
    all_y = [p[1] for line in lines for p in line]
    x_min_all, x_max_all = min(all_x), max(all_x)
    y_min_all, y_max_all = min(all_y), max(all_y)
    
    # Bestimme welcher Punkt der Referenzwand näher an der gewünschten Ecke liegt
    if ORIGIN_CORNER == 'NE':
        # Nordost: höchstes X und Y -> wähle Punkt mit höherem X auf der Referenzwand
        origin = p1 if p1[0] > p2[0] else p2
        corner_name = 'Nordost'
    elif ORIGIN_CORNER == 'NW':
        # Nordwest: niedrigstes X, höchstes Y -> wähle Punkt mit niedrigerem X
        origin = p1 if p1[0] < p2[0] else p2
        corner_name = 'Nordwest'
    elif ORIGIN_CORNER == 'SE':
        # Südost: höchstes X, niedrigstes Y -> wähle Punkt mit höherem X
        origin = p1 if p1[0] > p2[0] else p2
        corner_name = 'Südost'
    elif ORIGIN_CORNER == 'SW':
        # Südwest: niedrigstes X, niedrigstes Y -> wähle Punkt mit niedrigerem X
        origin = p1 if p1[0] < p2[0] else p2
        corner_name = 'Südwest'
    else:
        print(f"  ! Ungültige ORIGIN_CORNER: {ORIGIN_CORNER}. Verwende 'NE'")
        origin = p1 if p1[0] > p2[0] else p2
        corner_name = 'Nordost'
    
    # Winkel der Referenzwand zur X-Achse
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.arctan2(dy, dx)
    
    # Durchschnittliche Position der Referenzwand
    avg_x = (p1[0] + p2[0]) / 2
    avg_y = (p1[1] + p2[1]) / 2
    
    reference = {
        'ref_wall': ref_wall,
        'origin': origin,
        'angle': angle,
        'p1': p1,
        'p2': p2,
        'ref_coord': (avg_x, avg_y),  # Für Positionierung der Referenzlinie
        'wall_type': REFERENCE_WALL,
        'corner_type': ORIGIN_CORNER
    }
    
    print(f"  → Referenzwand: {wall_name_map.get(REFERENCE_WALL, 'Unbekannt')}")
    print(f"  → Nullpunkt-Ecke: {corner_name} X={origin[0]:.1f}, Y={origin[1]:.1f}")
    print(f"  → Referenzwand-Winkel: {np.degrees(angle):.2f}°")
    
    return reference


def calculate_deviations(points, radius):
    """Berechnet lokale Abweichungen für jeden Punkt"""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # KD-Tree für effiziente Nachbarsuche
    xy_coords = np.column_stack([x, y])
    tree = cKDTree(xy_coords)
    
    deviations = []
    for i in range(len(points)):
        neighbors = tree.query_ball_point(xy_coords[i], radius)
        if len(neighbors) >= 5:
            # Median der Nachbarn als lokale Referenz
            deviations.append(z[i] - np.median(z[neighbors]))
        else:
            deviations.append(np.nan)
    
    return np.array(deviations)


def create_png(points, deviations, grundriss, threshold, output_path, north_ref=None):
    """Erstellt hochauflösende PNG-Grafik"""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Wenn Nordwand-Referenz vorhanden, nutze diese für Nullpunkt
    if north_ref:
        origin = north_ref['origin']
        ref_coord = north_ref['ref_coord']
        wall_type = north_ref['wall_type']
    else:
        origin = (x_max, y_max)
        ref_coord = ((x_min + x_max)/2, y_max)
        wall_type = 'N'
    
    # Interpoliertes Raster für Heatmap
    grid_res = 50
    x_grid = np.arange(x_min, x_max, grid_res)
    y_grid = np.arange(y_min, y_max, grid_res)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    valid = ~np.isnan(deviations)
    Z_grid = griddata((x[valid], y[valid]), deviations[valid], (X_grid, Y_grid), method='linear')
    
    # Colormap
    vmax = 8
    colors_list = ['#0000AA', '#4444FF', '#8888FF', '#CCCCFF', '#FFFFFF', 
                   '#FFCCCC', '#FF8888', '#FF4444', '#AA0000']
    cmap = LinearSegmentedColormap.from_list('abtrag_spachtel', colors_list)
    
    # Statistiken
    abtragen = np.sum(deviations > threshold)
    spachteln = np.sum(deviations < -threshold)
    max_abtragen = np.nanmax(deviations)
    max_spachteln = abs(np.nanmin(deviations))
    
    # Plot erstellen
    fig, ax = plt.subplots(figsize=(30, 25))
    
    # Heatmap
    im = ax.pcolormesh(X_grid/1000, Y_grid/1000, Z_grid, cmap=cmap, 
                       vmin=-vmax, vmax=vmax, shading='auto', alpha=0.7)
    
    # Grundriss
    for start, end in grundriss:
        ax.plot([start[0]/1000, end[0]/1000], [start[1]/1000, end[1]/1000], 
                'k-', linewidth=4, zorder=20)
    
    # Referenzwand (Nordwand/Südwand/etc.)
    wall_names = {'N': 'NORDWAND', 'S': 'SÜDWAND', 'E': 'OSTWAND', 'W': 'WESTWAND'}
    wall_name = wall_names.get(wall_type, 'REFERENZWAND')
    
    if wall_type in ['N', 'S']:
        # Horizontale Linie
        ax.axhline(y=ref_coord[1]/1000, color='red', linewidth=5, zorder=21)
        ax.text(ref_coord[0]/1000, ref_coord[1]/1000 + 0.2, f'{wall_name} (Bezug/Nullpunkt)', 
                fontsize=16, ha='center', va='bottom', color='red', fontweight='bold')
    else:
        # Vertikale Linie
        ax.axvline(x=ref_coord[0]/1000, color='red', linewidth=5, zorder=21)
        ax.text(ref_coord[0]/1000 + 0.2, ref_coord[1]/1000, f'{wall_name} (Bezug/Nullpunkt)', 
                fontsize=16, ha='left', va='center', color='red', fontweight='bold', rotation=90)
    
    # Nullpunkt
    ax.plot(origin[0]/1000, origin[1]/1000, 'o', markersize=15, markerfacecolor='red', 
            markeredgecolor='darkred', markeredgewidth=2, zorder=25)
    ax.text(origin[0]/1000 + 0.08, origin[1]/1000 + 0.08, '0,0', fontsize=12, 
            color='red', fontweight='bold', zorder=25)
    
    # Bestimme Raster-Startpunkte für saubere 50cm/10cm Linien
    ref_y = ref_coord[1]
    
    # 10cm Raster - über gesamten Raum
    for yy in np.arange(y_min - 100, y_max + 200, 100):
        abstand = abs(ref_y - yy)
        is_500cm = (int(round(abstand)) % 500 == 0)
        if not is_500cm:
            ax.axhline(y=yy/1000, color='lightgray', linewidth=0.4, alpha=0.5, zorder=4)
    
    for xx in np.arange(x_min - 100, x_max + 200, 100):
        abstand = abs(origin[0] - xx)
        is_500cm = (int(round(abstand)) % 500 == 0)
        if not is_500cm:
            ax.axvline(x=xx/1000, color='lightgray', linewidth=0.4, alpha=0.5, zorder=4)
    
    # 50cm Raster mit Beschriftung - Y-Richtung (von Referenzwand)
    # Finde erste 50cm Linie und zeichne dann alle in beide Richtungen
    start_y = ref_y
    # Nach unten
    for mult in range(0, int((ref_y - y_min + 500) / 500) + 1):
        yy = ref_y - mult * 500
        if yy >= y_min - 100:
            ax.axhline(y=yy/1000, color='dimgray', linewidth=1.2, alpha=0.7, zorder=5)
            abstand = int(abs(ref_y - yy))
            if abstand > 0:
                ax.text(x_min/1000 - 0.12, yy/1000, f"{abstand}", 
                        fontsize=9, ha='right', va='center', color='gray')
                ax.text(x_max/1000 + 0.12, yy/1000, f"{abstand}", 
                        fontsize=9, ha='left', va='center', color='gray')
    # Nach oben
    for mult in range(1, int((y_max - ref_y + 500) / 500) + 1):
        yy = ref_y + mult * 500
        if yy <= y_max + 100:
            ax.axhline(y=yy/1000, color='dimgray', linewidth=1.2, alpha=0.7, zorder=5)
            abstand = int(abs(ref_y - yy))
            ax.text(x_min/1000 - 0.12, yy/1000, f"{abstand}", 
                    fontsize=9, ha='right', va='center', color='gray')
            ax.text(x_max/1000 + 0.12, yy/1000, f"{abstand}", 
                    fontsize=9, ha='left', va='center', color='gray')
    
    # 50cm Raster mit Beschriftung - X-Richtung (von Nullpunkt-Ecke)
    # Nach links
    for mult in range(0, int((origin[0] - x_min + 500) / 500) + 1):
        xx = origin[0] - mult * 500
        if xx >= x_min - 100:
            ax.axvline(x=xx/1000, color='dimgray', linewidth=1.2, alpha=0.7, zorder=5)
            abstand = int(abs(origin[0] - xx))
            if abstand > 0:
                ax.text(xx/1000, ref_y/1000 + 0.12, f"{abstand}", 
                        fontsize=9, ha='center', va='bottom', color='gray')
    # Nach rechts
    for mult in range(1, int((x_max - origin[0] + 500) / 500) + 1):
        xx = origin[0] + mult * 500
        if xx <= x_max + 100:
            ax.axvline(x=xx/1000, color='dimgray', linewidth=1.2, alpha=0.7, zorder=5)
            abstand = int(abs(origin[0] - xx))
            ax.text(xx/1000, ref_y/1000 + 0.12, f"{abstand}", 
                    fontsize=9, ha='center', va='bottom', color='gray')
    
    # Problemstellen markieren
    for i in range(len(points)):
        if np.isnan(deviations[i]):
            continue
        dev = deviations[i]
        px, py = x[i]/1000, y[i]/1000
        
        if dev > threshold:
            ax.plot(px, py, 'o', markersize=8, markerfacecolor='none', 
                    markeredgecolor='darkred', markeredgewidth=2, zorder=15)
            ax.text(px, py - 0.04, f'-{dev:.1f}', fontsize=7, ha='center', va='top',
                    color='darkred', fontweight='bold', zorder=16)
        elif dev < -threshold:
            ax.plot(px, py, 's', markersize=8, markerfacecolor='none',
                    markeredgecolor='darkblue', markeredgewidth=2, zorder=15)
            ax.text(px, py - 0.04, f'+{abs(dev):.1f}', fontsize=7, ha='center', va='top',
                    color='darkblue', fontweight='bold', zorder=16)
    
    # Bemaßung
    breite_m = (x_max - x_min) / 1000
    tiefe_m = (y_max - y_min) / 1000
    ax.annotate('', xy=(x_max/1000, y_min/1000 - 0.3), xytext=(x_min/1000, y_min/1000 - 0.3),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text((x_min + x_max)/2/1000, y_min/1000 - 0.4, f'{breite_m:.2f} m', 
            fontsize=14, ha='center', va='top', fontweight='bold')
    ax.annotate('', xy=(x_min/1000 - 0.3, y_max/1000), xytext=(x_min/1000 - 0.3, y_min/1000),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(x_min/1000 - 0.4, (y_min + y_max)/2/1000, f'{tiefe_m:.2f} m', 
            fontsize=14, ha='right', va='center', fontweight='bold', rotation=90)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label('Abweichung [mm]\n← Spachteln | Abtragen →', fontsize=12)
    
    # Legende
    corner_names = {'NE': 'Nordost', 'NW': 'Nordwest', 'SE': 'Südost', 'SW': 'Südwest'}
    corner_name = corner_names.get(north_ref['corner_type'] if north_ref else 'NE', 'Nordost')
    
    legend_text = (f"AUSGLEICHSPLAN (Schwelle ±{threshold}mm)\n"
                   f"Nullpunkt: {corner_name}-Ecke an {wall_name} (0,0)\n"
                   f"Maße in mm von Referenzwand\n\n"
                   f"○ ROT = ABTRAGEN (Wert = wieviel mm abschleifen)\n"
                   f"□ BLAU = SPACHTELN (Wert = wieviel mm Masse auftragen)\n\n"
                   f"Abtragen: {abtragen} Stellen (max. {max_abtragen:.1f}mm)\n"
                   f"Spachteln: {spachteln} Stellen (max. {max_spachteln:.1f}mm)")
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_title(f'BODENAUSGLEICHSPLAN\nMaße von {wall_name} | Raster: dünn=10cm, dick=50cm | Nullpunkt: {corner_name}-Ecke',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlim(x_min/1000 - 0.5, x_max/1000 + 0.3)
    ax.set_ylim(y_min/1000 - 0.5, y_max/1000 + 0.4)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return abtragen, spachteln, max_abtragen, max_spachteln


def create_dxf(points, deviations, grundriss, threshold, output_path, north_ref=None):
    """Erstellt DXF mit Layern"""
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Wenn Nordwand-Referenz vorhanden, nutze diese für Nullpunkt
    if north_ref:
        origin = north_ref['origin']
        ref_coord = north_ref['ref_coord']
        wall_type = north_ref['wall_type']
    else:
        origin = (x_max, y_max)
        ref_coord = ((x_min + x_max)/2, y_max)
        wall_type = 'N'
    
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Layer
    doc.layers.add('GRUNDRISS', color=7)
    doc.layers.add('NORDWAND', color=1)
    doc.layers.add('RASTER_500', color=8)
    doc.layers.add('RASTER_100', color=9)
    doc.layers.add('BESCHRIFTUNG', color=8)
    doc.layers.add('ABTRAGEN_MARKER', color=1)
    doc.layers.add('ABTRAGEN_WERT', color=1)
    doc.layers.add('SPACHTELN_MARKER', color=5)
    doc.layers.add('SPACHTELN_WERT', color=5)
    doc.layers.add('LEGENDE', color=7)
    
    # Grundriss
    for start, end in grundriss:
        msp.add_line(start, end, dxfattribs={'layer': 'GRUNDRISS', 'lineweight': 70})
    
    # Referenzwand
    wall_names = {'N': 'NORDWAND', 'S': 'SÜDWAND', 'E': 'OSTWAND', 'W': 'WESTWAND'}
    wall_name = wall_names.get(wall_type, 'REFERENZWAND')
    
    if wall_type in ['N', 'S']:
        msp.add_line((x_min - 300, ref_coord[1]), (x_max + 300, ref_coord[1]),
                     dxfattribs={'layer': 'NORDWAND', 'lineweight': 100})
        msp.add_text(f"{wall_name} (Bezug/Nullpunkt)", height=120,
                     dxfattribs={'layer': 'NORDWAND'}
        ).set_placement((ref_coord[0], ref_coord[1] + 200), align=TextEntityAlignment.CENTER)
    else:
        msp.add_line((ref_coord[0], y_min - 300), (ref_coord[0], y_max + 300),
                     dxfattribs={'layer': 'NORDWAND', 'lineweight': 100})
        msp.add_text(f"{wall_name} (Bezug/Nullpunkt)", height=120,
                     dxfattribs={'layer': 'NORDWAND'}
        ).set_placement((ref_coord[0] + 200, ref_coord[1]), align=TextEntityAlignment.LEFT)
    
    # 10cm und 50cm Raster - vereinfachte Version
    # Erzeuge Raster über gesamten Raum, Beschriftung nur an 50cm Linien
    for yy in np.arange(y_min, y_max + 100, 100):
        is_500 = (int(abs(ref_coord[1] - yy)) % 500 == 0)
        if is_500:
            msp.add_line((x_min - 100, yy), (x_max + 100, yy),
                         dxfattribs={'layer': 'RASTER_500', 'lineweight': 30})
            abstand = int(abs(ref_coord[1] - yy))
            if abstand > 0:  # Nicht Nulllinie beschriften
                msp.add_text(f"{abstand}", height=60,
                             dxfattribs={'layer': 'BESCHRIFTUNG'}
                ).set_placement((x_min - 150, yy), align=TextEntityAlignment.RIGHT)
                msp.add_text(f"{abstand}", height=60,
                             dxfattribs={'layer': 'BESCHRIFTUNG'}
                ).set_placement((x_max + 150, yy), align=TextEntityAlignment.LEFT)
        else:
            msp.add_line((x_min, yy), (x_max, yy),
                         dxfattribs={'layer': 'RASTER_100', 'lineweight': 9})
    
    for xx in np.arange(x_min, x_max + 100, 100):
        is_500 = (int(abs(origin[0] - xx)) % 500 == 0)
        if is_500:
            msp.add_line((xx, y_min - 100), (xx, y_max + 100),
                         dxfattribs={'layer': 'RASTER_500', 'lineweight': 30})
            abstand = int(abs(origin[0] - xx))
            if abstand > 0:  # Nicht Nulllinie beschriften
                msp.add_text(f"{abstand}", height=60,
                             dxfattribs={'layer': 'BESCHRIFTUNG'}
                ).set_placement((xx, ref_coord[1] + 200), align=TextEntityAlignment.CENTER)
        else:
            msp.add_line((xx, y_min), (xx, y_max),
                         dxfattribs={'layer': 'RASTER_100', 'lineweight': 9})
    
    # Problemstellen
    dxf_abtragen = 0
    dxf_spachteln = 0
    max_abtragen = 0
    max_spachteln = 0
    
    for i in range(len(points)):
        if np.isnan(deviations[i]):
            continue
        dev = deviations[i]
        px, py = x[i], y[i]
        
        if dev > threshold:
            r = max(30, abs(dev) * 10)
            msp.add_circle((px, py), r, dxfattribs={'layer': 'ABTRAGEN_MARKER'})
            msp.add_text(f"-{dev:.1f}", height=45,
                         dxfattribs={'layer': 'ABTRAGEN_WERT'}
            ).set_placement((px, py - r - 20), align=TextEntityAlignment.CENTER)
            dxf_abtragen += 1
            max_abtragen = max(max_abtragen, dev)
        elif dev < -threshold:
            r = max(30, abs(dev) * 10)
            msp.add_lwpolyline([(px-r, py-r), (px+r, py-r), (px+r, py+r), (px-r, py+r), (px-r, py-r)],
                               dxfattribs={'layer': 'SPACHTELN_MARKER'})
            msp.add_text(f"+{abs(dev):.1f}", height=45,
                         dxfattribs={'layer': 'SPACHTELN_WERT'}
            ).set_placement((px, py - r - 20), align=TextEntityAlignment.CENTER)
            dxf_spachteln += 1
            max_spachteln = max(max_spachteln, abs(dev))
    
    # Legende
    lx, ly = x_min - 600, y_max + 600
    msp.add_text(f"AUSGLEICHSPLAN (Schwelle ±{threshold}mm) | Raster 10cm/50cm", height=120,
                 dxfattribs={'layer': 'LEGENDE'}
    ).set_placement((lx, ly), align=TextEntityAlignment.LEFT)
    
    msp.add_circle((lx + 50, ly - 200), 40, dxfattribs={'layer': 'ABTRAGEN_MARKER'})
    msp.add_text(f"ABTRAGEN: {dxf_abtragen}x (max. {max_abtragen:.1f}mm)", height=80,
                 dxfattribs={'layer': 'ABTRAGEN_WERT'}
    ).set_placement((lx + 150, ly - 200), align=TextEntityAlignment.LEFT)
    
    msp.add_lwpolyline([(lx+10, ly-320), (lx+90, ly-320), (lx+90, ly-400), (lx+10, ly-400), (lx+10, ly-320)],
                       dxfattribs={'layer': 'SPACHTELN_MARKER'})
    msp.add_text(f"SPACHTELN: {dxf_spachteln}x (max. {max_spachteln:.1f}mm)", height=80,
                 dxfattribs={'layer': 'SPACHTELN_WERT'}
    ).set_placement((lx + 150, ly - 360), align=TextEntityAlignment.LEFT)
    
    # Nullpunkt
    msp.add_circle((origin[0], origin[1]), 80, dxfattribs={'layer': 'NORDWAND'})
    msp.add_text("0,0", height=70, dxfattribs={'layer': 'NORDWAND'}
    ).set_placement((origin[0] + 120, origin[1] + 120), align=TextEntityAlignment.LEFT)
    
    doc.saveas(output_path)


def main():
    """Hauptfunktion"""
    print("=" * 60)
    print("BODENANALYSE - Ausgleichsplan Generator")
    print("=" * 60)
    
    # Abhängigkeiten prüfen
    check_dependencies()
    
    # Ausgabeverzeichnis erstellen
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # DXF laden
    print(f"\n[1/5] Lade DXF-Datei...")
    points, grundriss = load_dxf(INPUT_DXF)
    
    # Nordwand finden (für Referenz-Beschriftung)
    print(f"\n[2/5] Finde Referenzwand für Koordinatensystem...")
    print(f"      Einstellung: REFERENCE_WALL={REFERENCE_WALL}, ORIGIN_CORNER={ORIGIN_CORNER}")
    north_ref = find_north_wall(grundriss)
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    print(f"\nRaumgröße: {(x.max()-x.min())/1000:.2f} x {(y.max()-y.min())/1000:.2f} m")
    print(f"Höhenbereich: {z.min():.1f} bis {z.max():.1f} mm (Δ {z.max()-z.min():.1f} mm)")
    
    # Abweichungen berechnen
    print(f"\n[3/5] Berechne lokale Abweichungen (Radius: {RADIUS}mm)...")
    deviations = calculate_deviations(points, RADIUS)
    
    abtragen = np.sum(deviations > SCHWELLE)
    spachteln = np.sum(deviations < -SCHWELLE)
    print(f"\nErgebnis (Schwelle ±{SCHWELLE}mm):")
    print(f"  Abtragen:  {abtragen} Stellen (max. {np.nanmax(deviations):.1f}mm)")
    print(f"  Spachteln: {spachteln} Stellen (max. {abs(np.nanmin(deviations)):.1f}mm)")
    
    # Ausgaben erstellen
    print(f"\n[4/5] Erstelle Ausgabedateien...")
    
    if EXPORT_PNG:
        png_path = os.path.join(OUTPUT_DIR, f"ausgleichsplan_{SCHWELLE}mm.png")
        create_png(points, deviations, grundriss, SCHWELLE, png_path, north_ref)
        print(f"  ✓ {png_path}")
    
    if EXPORT_PDF:
        pdf_path = os.path.join(OUTPUT_DIR, f"ausgleichsplan_{SCHWELLE}mm.pdf")
        # PDF ist identisch mit PNG, nur anderes Format
        fig = plt.figure()
        create_png(points, deviations, grundriss, SCHWELLE, pdf_path, north_ref)
        print(f"  ✓ {pdf_path}")
    
    if EXPORT_DXF:
        dxf_path = os.path.join(OUTPUT_DIR, f"ausgleichsplan_{SCHWELLE}mm.dxf")
        create_dxf(points, deviations, grundriss, SCHWELLE, dxf_path, north_ref)
        print(f"  ✓ {dxf_path}")
    
    # CSV mit allen Punkten
    print(f"\n[5/5] Exportiere CSV...")
    csv_path = os.path.join(OUTPUT_DIR, f"alle_punkte_{SCHWELLE}mm.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('X [m];Y [m];Ist-Höhe [mm];Abweichung [mm];Aktion\n')
        for i in range(len(points)):
            if not np.isnan(deviations[i]):
                dev = deviations[i]
                if dev > SCHWELLE:
                    aktion = "ABTRAGEN"
                elif dev < -SCHWELLE:
                    aktion = "SPACHTELN"
                else:
                    aktion = "OK"
                f.write(f"{x[i]/1000:.3f};{y[i]/1000:.3f};{z[i]:.1f};{dev:.1f};{aktion}\n")
    print(f"  ✓ {csv_path}")
    
    print(f"\n" + "=" * 60)
    print("FERTIG!")
    print(f"Ausgabeverzeichnis: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
