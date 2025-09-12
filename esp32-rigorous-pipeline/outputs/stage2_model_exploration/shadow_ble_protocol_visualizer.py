#!/usr/bin/env python3
"""
Shadow BLE Protocol Visualizer & RFC Generator

This script creates comprehensive visualizations and documentation for the Shadow BLE protocol
based on analysis of the Swift macOS implementation and ESP32 firmware.

Components:
1. BLE Protocol State Machine Diagram
2. Data Packet Format Specifications  
3. Communication Flow Sequence Diagrams
4. RFC-style Protocol Documentation

Author: Ashidu Dissanayake
Date: December 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle, Circle, FancyArrowPatch
import numpy as np
from pathlib import Path
import matplotlib.gridspec as gridspec
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8')

class ShadowBLEProtocolVisualizer:
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else Path('./ble_protocol_visualizations')
        self.output_dir.mkdir(exist_ok=True)
        
        # Color scheme for protocol components
        self.colors = {
            'esp32': '#00a86b',
            'macos': '#007aff', 
            'ble': '#ff6b35',
            'data': '#ffd23f',
            'error': '#e74c3c',
            'success': '#27ae60',
            'neutral': '#95a5a6'
        }
        
    def plot_ble_state_machine(self):
        """Create BLE protocol state machine diagram"""
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Define states and their positions
        states = {
            'idle': {'pos': (2, 8), 'color': self.colors['neutral']},
            'advertising': {'pos': (6, 8), 'color': self.colors['esp32']},
            'scanning': {'pos': (2, 5), 'color': self.colors['macos']},
            'connecting': {'pos': (6, 5), 'color': self.colors['ble']},
            'connected': {'pos': (10, 5), 'color': self.colors['success']},
            'requesting': {'pos': (6, 2), 'color': self.colors['data']},
            'syncing': {'pos': (10, 2), 'color': self.colors['data']},
            'error': {'pos': (12, 8), 'color': self.colors['error']}
        }
        
        # Draw states
        for state_name, info in states.items():
            x, y = info['pos']
            
            # State circle
            circle = Circle((x, y), 0.8, facecolor=info['color'], 
                          edgecolor='black', alpha=0.8, linewidth=2)
            ax.add_patch(circle)
            
            # State label
            ax.text(x, y, state_name.replace('_', '\n').title(), 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   color='white' if state_name != 'idle' else 'black')
        
        # Define transitions
        transitions = [
            # ESP32 side
            ('idle', 'advertising', 'ESP32 Boot'),
            ('advertising', 'connected', 'macOS Connect'),
            ('connected', 'requesting', 'Read Request'),
            ('requesting', 'syncing', 'Send Data'),
            ('syncing', 'advertising', 'Disconnect'),
            
            # macOS side  
            ('idle', 'scanning', 'BLE Start'),
            ('scanning', 'connecting', 'Device Found'),
            ('connecting', 'connected', 'Connection OK'),
            ('connected', 'requesting', 'Request Missed'),
            
            # Error handling
            ('connecting', 'error', 'Connect Fail'),
            ('connected', 'error', 'Protocol Error'),
            ('error', 'scanning', 'Retry'),
            ('advertising', 'idle', 'Timeout')
        ]
        
        # Draw transitions
        for start, end, label in transitions:
            start_pos = states[start]['pos']
            end_pos = states[end]['pos']
            
            # Calculate arrow position (offset from circle edge)
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                # Normalize and offset from circle edge
                offset = 0.9
                start_offset = (start_pos[0] + offset * dx/length, 
                               start_pos[1] + offset * dy/length)
                end_offset = (end_pos[0] - offset * dx/length,
                             end_pos[1] - offset * dy/length)
                
                # Draw arrow
                arrow = FancyArrowPatch(start_offset, end_offset,
                                      arrowstyle='->', mutation_scale=15,
                                      color='black', alpha=0.7, linewidth=1.5)
                ax.add_patch(arrow)
                
                # Add label
                mid_x = (start_offset[0] + end_offset[0]) / 2
                mid_y = (start_offset[1] + end_offset[1]) / 2
                
                # Adjust label position to avoid overlap
                angle = np.arctan2(dy, dx) * 180 / np.pi
                if abs(angle) > 90:
                    angle += 180
                    
                ax.text(mid_x, mid_y + 0.3, label, ha='center', va='center',
                       fontsize=8, rotation=angle if abs(angle) < 45 else 0,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Add protocol details box
        protocol_box = FancyBboxPatch((0.5, 0.5), 6, 2,
                                     boxstyle="round,pad=0.1",
                                     facecolor='lightblue', alpha=0.7)
        ax.add_patch(protocol_box)
        
        protocol_text = """Shadow BLE Protocol V1.0
Service UUID: A000
Event Characteristic: A002
Ring Buffer: 32 entries (7-bit sequence)
Reset Opcode: 0xFF (Magic: 0x52)
Data Format: State transitions with timestamps"""
        
        ax.text(3.5, 1.5, protocol_text, ha='center', va='center',
                fontsize=10, fontweight='bold')
        
        plt.title('Shadow BLE Protocol: State Machine Diagram', 
                 fontsize=16, fontweight='bold', pad=20)
        
        return fig
    
    def plot_packet_format_diagram(self):
        """Create detailed packet format diagrams"""
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # 1. Advertisement Packet Format
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_xlim(0, 16)
        ax1.set_ylim(0, 4)
        ax1.axis('off')
        
        # Advertisement packet structure
        packet_fields = [
            {'name': 'Sequence\n(7 bits)', 'start': 1, 'width': 3, 'color': '#3498db'},
            {'name': 'State\n(1 bit)', 'start': 4, 'width': 1, 'color': '#e74c3c'},
        ]
        
        for field in packet_fields:
            rect = Rectangle((field['start'], 1), field['width'], 1,
                           facecolor=field['color'], alpha=0.8, edgecolor='black')
            ax1.add_patch(rect)
            ax1.text(field['start'] + field['width']/2, 1.5, field['name'],
                    ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Bit numbering
        for i in range(8):
            ax1.text(1 + i*0.5, 0.5, str(7-i), ha='center', va='center', fontsize=8)
            ax1.axvline(x=1+i*0.5, ymin=0.25, ymax=0.75, color='gray', alpha=0.5)
        
        ax1.text(3, 3, 'Advertisement Packet (1 byte)', ha='center', va='center',
                fontsize=14, fontweight='bold')
        ax1.text(3, 0.2, 'Bit Layout: SSSSSSS S (7-bit sequence + 1-bit state)',
                ha='center', va='center', fontsize=10, style='italic')
        
        # 2. Reset Request/Response
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_xlim(0, 8)
        ax2.set_ylim(0, 6)
        ax2.axis('off')
        
        # Reset request
        reset_req = Rectangle((1, 4), 6, 1, facecolor='#ff6b35', alpha=0.8, edgecolor='black')
        ax2.add_patch(reset_req)
        ax2.text(4, 4.5, 'Reset Request: 0xFF', ha='center', va='center', 
                fontweight='bold', color='white')
        
        # Reset response
        reset_resp_fields = [
            {'name': '0x00', 'start': 1, 'width': 1.5, 'color': '#95a5a6'},
            {'name': 'State', 'start': 2.5, 'width': 1.5, 'color': '#e74c3c'},
            {'name': '0x00', 'start': 4, 'width': 1.5, 'color': '#95a5a6'},
            {'name': '0x52', 'start': 5.5, 'width': 1.5, 'color': '#f39c12'}
        ]
        
        for field in reset_resp_fields:
            rect = Rectangle((field['start'], 2), field['width'], 1,
                           facecolor=field['color'], alpha=0.8, edgecolor='black')
            ax2.add_patch(rect)
            ax2.text(field['start'] + field['width']/2, 2.5, field['name'],
                    ha='center', va='center', fontweight='bold', fontsize=9)
        
        ax2.text(4, 1, 'Reset Response (4 bytes)', ha='center', va='center',
                fontsize=12, fontweight='bold')
        ax2.text(4, 0.3, 'Magic: 0x52 confirms reset', ha='center', va='center',
                fontsize=10, style='italic')
        
        # 3. Missed Data Request/Response
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_xlim(0, 8)
        ax3.set_ylim(0, 6)
        ax3.axis('off')
        
        # Missed request
        missed_req = Rectangle((1, 4), 6, 1, facecolor='#9b59b6', alpha=0.8, edgecolor='black')
        ax3.add_patch(missed_req)
        ax3.text(4, 4.5, 'Missed Request: Last Known Seq', ha='center', va='center',
                fontweight='bold', color='white')
        
        # Minimal response (2 bytes)
        minimal_fields = [
            {'name': 'Current\nSeq', 'start': 1, 'width': 3, 'color': '#3498db'},
            {'name': 'Current\nState', 'start': 4, 'width': 3, 'color': '#e74c3c'}
        ]
        
        for field in minimal_fields:
            rect = Rectangle((field['start'], 2), field['width'], 1,
                           facecolor=field['color'], alpha=0.8, edgecolor='black')
            ax3.add_patch(rect)
            ax3.text(field['start'] + field['width']/2, 2.5, field['name'],
                    ha='center', va='center', fontweight='bold', fontsize=9)
        
        ax3.text(4, 1, 'Minimal Response (2 bytes)', ha='center', va='center',
                fontsize=12, fontweight='bold')
        ax3.text(4, 0.3, 'When delta = 1', ha='center', va='center',
                fontsize=10, style='italic')
        
        # 4. Extended Response Format
        ax4 = fig.add_subplot(gs[2, :])
        ax4.set_xlim(0, 16)
        ax4.set_ylim(0, 4)
        ax4.axis('off')
        
        extended_fields = [
            {'name': 'Current\nSeq', 'start': 1, 'width': 2, 'color': '#3498db'},
            {'name': 'Current\nState', 'start': 3, 'width': 2, 'color': '#e74c3c'},
            {'name': 'Missed\nCount', 'start': 5, 'width': 2, 'color': '#f39c12'},
            {'name': 'Missed Entry 1\n(Seq + State)', 'start': 7, 'width': 3, 'color': '#2ecc71'},
            {'name': 'Missed Entry N\n(Seq + State)', 'start': 10, 'width': 3, 'color': '#2ecc71'},
            {'name': '...', 'start': 13, 'width': 2, 'color': '#95a5a6'}
        ]
        
        for field in extended_fields:
            rect = Rectangle((field['start'], 1.5), field['width'], 1,
                           facecolor=field['color'], alpha=0.8, edgecolor='black')
            ax4.add_patch(rect)
            ax4.text(field['start'] + field['width']/2, 2, field['name'],
                    ha='center', va='center', fontweight='bold', fontsize=9)
        
        ax4.text(8, 3.5, 'Extended Response (Variable Length)', ha='center', va='center',
                fontsize=14, fontweight='bold')
        ax4.text(8, 0.8, 'Format: Current + Count + [Missed Entries...] (when delta > 1)',
                ha='center', va='center', fontsize=10, style='italic')
        
        plt.suptitle('Shadow BLE Protocol: Packet Format Specifications', 
                    fontsize=16, fontweight='bold')
        
        return fig
    
    def plot_communication_sequence(self):
        """Create communication sequence diagram"""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 16)
        ax.axis('off')
        
        # Actors
        actors = [
            {'name': 'ESP32\nShadow Device', 'x': 2, 'color': self.colors['esp32']},
            {'name': 'BLE\nAdvertisement', 'x': 5, 'color': self.colors['ble']},
            {'name': 'macOS\nShadow App', 'x': 8, 'color': self.colors['macos']},
            {'name': 'Core Data\nRepository', 'x': 11, 'color': self.colors['data']}
        ]
        
        # Draw actor columns
        for actor in actors:
            # Actor box
            actor_box = FancyBboxPatch((actor['x']-0.8, 14.5), 1.6, 1,
                                      boxstyle="round,pad=0.1",
                                      facecolor=actor['color'], alpha=0.8)
            ax.add_patch(actor_box)
            ax.text(actor['x'], 15, actor['name'], ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')
            
            # Lifeline
            ax.axvline(x=actor['x'], ymin=0.05, ymax=0.9, color='gray', 
                      linestyle='--', alpha=0.7, linewidth=2)
        
        # Sequence steps
        steps = [
            # Step 1: Device startup
            {'from': 2, 'to': 5, 'y': 13, 'label': '1. Start Advertisement\n(seq=0, state=0)', 'type': 'request'},
            
            # Step 2: App scanning
            {'from': 8, 'to': 5, 'y': 12, 'label': '2. Scan for "Shadow"\nService A000', 'type': 'request'},
            
            # Step 3: Advertisement detected
            {'from': 5, 'to': 8, 'y': 11, 'label': '3. Advertisement Data\n(seq=5, state=1)', 'type': 'response'},
            
            # Step 4: Connection
            {'from': 8, 'to': 2, 'y': 10, 'label': '4. BLE Connect\nDiscover Characteristics', 'type': 'request'},
            
            # Step 5: Request missed data
            {'from': 8, 'to': 2, 'y': 9, 'label': '5. Write: Last Known Seq\n(req=2)', 'type': 'request'},
            
            # Step 6: Extended response
            {'from': 2, 'to': 8, 'y': 8, 'label': '6. Read: Extended Response\n(cur=5, state=1, missed=2)', 'type': 'response'},
            
            # Step 7: Data persistence
            {'from': 8, 'to': 11, 'y': 7, 'label': '7. Persist Transitions\n(seq 3,4,5)', 'type': 'request'},
            
            # Step 8: Disconnect
            {'from': 8, 'to': 2, 'y': 6, 'label': '8. Disconnect\nReturn to advertising', 'type': 'request'},
            
            # Step 9: Next advertisement cycle
            {'from': 2, 'to': 5, 'y': 5, 'label': '9. Continue Advertisement\n(seq=6, state=0)', 'type': 'request'},
            
            # Step 10: State change detection
            {'from': 2, 'to': 5, 'y': 4, 'label': '10. State Change\n(seq=7, state=1)', 'type': 'notification'},
            
            # Step 11: Immediate connection
            {'from': 5, 'to': 8, 'y': 3, 'label': '11. Delta=2 Detected\nInitiate Connection', 'type': 'notification'},
            
            # Step 12: Sync completion
            {'from': 8, 'to': 11, 'y': 2, 'label': '12. Sync Complete\nUp to date', 'type': 'success'}
        ]
        
        # Draw sequence arrows
        for step in steps:
            # Arrow style based on type
            if step['type'] == 'request':
                arrow_style = '->'
                color = 'blue'
            elif step['type'] == 'response':
                arrow_style = '<-'
                color = 'green'
            elif step['type'] == 'notification':
                arrow_style = '->'
                color = 'orange'
            else:  # success
                arrow_style = '->'
                color = 'purple'
            
            # Draw arrow
            arrow = FancyArrowPatch((step['from'], step['y']), (step['to'], step['y']),
                                  arrowstyle=arrow_style, mutation_scale=15,
                                  color=color, linewidth=2)
            ax.add_patch(arrow)
            
            # Add label
            mid_x = (step['from'] + step['to']) / 2
            ax.text(mid_x, step['y'] + 0.3, step['label'], ha='center', va='bottom',
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                                        facecolor='white', alpha=0.8))
        
        # Add timing annotations
        ax.text(0.5, 8, 'Real-time\nSynchronization\nPhase', ha='center', va='center',
               fontsize=10, fontweight='bold', rotation=90,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        ax.text(0.5, 4, 'Continuous\nMonitoring\nPhase', ha='center', va='center',
               fontsize=10, fontweight='bold', rotation=90,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        
        plt.title('Shadow BLE Protocol: Communication Sequence Flow', 
                 fontsize=16, fontweight='bold', pad=20)
        
        return fig
    
    def plot_neural_network_architecture(self):
        """Create detailed neural network architecture diagram"""
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Color scheme for different components
        colors = {
            'sensor': '#e74c3c',      # Red for sensors
            'preprocessing': '#f39c12', # Orange for preprocessing
            'features': '#3498db',     # Blue for features
            'ml': '#2ecc71',          # Green for ML
            'output': '#9b59b6',      # Purple for output
            'transmission': '#34495e'  # Dark for transmission
        }
        
        # 1. Sensor Layer (Left side)
        sensors = [
            {'name': 'MAX30105 (BVP)', 'pos': (1, 8), 'size': 0.3},
            {'name': 'MPU6050 (Accelerometer)', 'pos': (1, 6.5), 'size': 0.3},
            {'name': 'GSR/EDA Sensor', 'pos': (1, 5), 'size': 0.3},
            {'name': 'Temperature Sensor', 'pos': (1, 3.5), 'size': 0.3}
        ]
        
        for sensor in sensors:
            circle = plt.Circle(sensor['pos'], sensor['size'], 
                              color=colors['sensor'], alpha=0.8)
            ax.add_patch(circle)
            ax.text(sensor['pos'][0], sensor['pos'][1], sensor['name'], 
                   ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        # 2. Preprocessing Layer
        preprocess_box = FancyBboxPatch((2.5, 2), 2, 6,
                                       boxstyle="round,pad=0.1",
                                       facecolor=colors['preprocessing'],
                                       edgecolor='black', alpha=0.8)
        ax.add_patch(preprocess_box)
        ax.text(3.5, 7.5, 'Data Preprocessing', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
        
        # Preprocessing steps
        preprocess_steps = [
            'Circular Buffer (60s window)',
            'Sampling Rate Normalization',
            'Noise Filtering & Calibration',
            'Feature Extraction'
        ]
        
        for i, step in enumerate(preprocess_steps):
            step_y = 6.8 - i * 1.2
            step_box = FancyBboxPatch((2.7, step_y - 0.3), 1.6, 0.6,
                                     boxstyle="round,pad=0.05",
                                     facecolor=colors['preprocessing'], alpha=0.6)
            ax.add_patch(step_box)
            ax.text(3.5, step_y, step, ha='center', va='center',
                   fontsize=8, color='white')
        
        # 3. Feature Vector (30 features)
        feature_box = FancyBboxPatch((5.5, 3), 1.5, 4,
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['features'],
                                    edgecolor='black', alpha=0.8)
        ax.add_patch(feature_box)
        ax.text(6.25, 6.5, '30 Features', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
        
        # Feature categories
        feature_categories = [
            'BVP Features (8 features)',
            'ACC Features (12 features)', 
            'EDA Features (6 features)',
            'TEMP Features (4 features)'
        ]
        
        for i, category in enumerate(feature_categories):
            cat_y = 6 - i * 0.8
            ax.text(6.25, cat_y, category, ha='center', va='center',
                   fontsize=8, color='white')
        
        # 4. Neural Network Architecture
        nn_layers = [
            {'name': 'Input Layer', 'neurons': 30, 'pos': (8, 5), 'color': colors['ml']},
            {'name': 'Hidden Layer 1', 'neurons': 64, 'pos': (9.5, 5), 'color': colors['ml']},
            {'name': 'Hidden Layer 2', 'neurons': 32, 'pos': (11, 5), 'color': colors['ml']},
            {'name': 'Output Layer', 'neurons': 1, 'pos': (12.5, 5), 'color': colors['output']}
        ]
        
        # Draw neural network layers
        for layer in nn_layers:
            # Calculate node positions
            if layer['neurons'] <= 10:
                y_positions = np.linspace(3, 7, layer['neurons'])
            else:
                # For large layers, show representative nodes
                y_positions = np.linspace(3.5, 6.5, min(8, layer['neurons']))
            
            # Draw nodes
            for i, y in enumerate(y_positions):
                if layer['neurons'] > 10 and i == 4:
                    # Add "..." for large layers
                    ax.text(layer['pos'][0], y, '⋮', ha='center', va='center', 
                           fontsize=16, fontweight='bold')
                else:
                    circle = plt.Circle((layer['pos'][0], y), 0.1, 
                                      color=layer['color'], alpha=0.8)
                    ax.add_patch(circle)
            
            # Layer label
            ax.text(layer['pos'][0], 2.5, layer['name'], ha='center', va='center',
                   fontsize=10, fontweight='bold')
            ax.text(layer['pos'][0], 2.2, f'{layer["neurons"]} nodes', 
                   ha='center', va='center', fontsize=8, style='italic')
        
        # Draw connections between layers
        for i in range(len(nn_layers) - 1):
            curr_layer = nn_layers[i]
            next_layer = nn_layers[i + 1]
            
            # Draw sample connections
            for y1 in np.linspace(4, 6, 3):
                for y2 in np.linspace(4, 6, 3):
                    line = plt.Line2D([curr_layer['pos'][0] + 0.1, next_layer['pos'][0] - 0.1],
                                    [y1, y2], color='gray', alpha=0.3, linewidth=1)
                    ax.add_line(line)
        
        # 5. BLE Transmission Layer
        ble_box = FancyBboxPatch((8, 0.5), 4.5, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['transmission'],
                                edgecolor='black', alpha=0.8)
        ax.add_patch(ble_box)
        ax.text(10.25, 1.25, 'BLE Transmission Pipeline', 
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        # BLE components
        ble_components = ['Stress Prediction', 'Data Packaging', 'BLE Transmission']
        for i, comp in enumerate(ble_components):
            comp_box = FancyBboxPatch((8.2 + i * 1.4, 0.7), 1.2, 0.6,
                                     boxstyle="round,pad=0.05",
                                     facecolor=colors['transmission'], alpha=0.6)
            ax.add_patch(comp_box)
            ax.text(8.8 + i * 1.4, 1, comp, ha='center', va='center',
                   fontsize=8, color='white')
        
        # Add data flow arrows
        flow_arrows = [
            # Sensors to preprocessing
            ((1.3, 6), (2.5, 6)),
            # Preprocessing to features
            ((4.5, 5), (5.5, 5)),
            # Features to ML
            ((7, 5), (8, 5)),
            # ML output to BLE
            ((12.5, 4.5), (10.25, 2))
        ]
        
        for start, end in flow_arrows:
            arrow = ConnectionPatch(start, end, "data", "data",
                                  arrowstyle="->", shrinkA=5, shrinkB=5,
                                  mutation_scale=15, fc="black", lw=2)
            ax.add_patch(arrow)
        
        # Add technical specifications
        specs_text = """
        Technical Specifications:
        • Sampling: 16.67Hz (60s windows)
        • Features: 30 physiological signals
        • Model: MLP (30→64→32→1)
        • Inference: ~3.8ms per prediction
        • BLE: Custom stress monitoring protocol
        • Power: <50mA average consumption
        """
        
        ax.text(0.5, 9.5, specs_text, ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # Add activation functions
        activations = [
            {'x': 8.75, 'text': 'ReLU'},
            {'x': 10.25, 'text': 'ReLU'},
            {'x': 11.75, 'text': 'Sigmoid'}
        ]
        
        for act in activations:
            ax.text(act['x'], 7.5, act['text'], ha='center', va='center',
                   fontsize=10, fontweight='bold', color='#9b59b6',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        plt.title('Shadow: Complete Neural Network & Data Pipeline Architecture', 
                 fontsize=16, fontweight='bold', pad=20)
        
        return fig

    def generate_rfc_document(self):
        """Generate RFC-style protocol documentation"""
        rfc_path = self.output_dir / 'shadow_ble_protocol_rfc.md'
        
        with open(rfc_path, 'w') as f:
            f.write("""# Shadow BLE Protocol Specification (RFC-Style)

**Document:** Shadow-BLE-001  
**Version:** 1.0  
**Date:** December 2025  
**Author:** Ashidu Dissanayake  
**Status:** Implementation Draft

---

## Abstract

This document specifies the Bluetooth Low Energy (BLE) communication protocol used by the Shadow stress detection system. The protocol enables efficient synchronization of stress state transitions between ESP32-based wearable devices and macOS client applications with support for data loss recovery and ring buffer management.

---

## 1. Introduction

### 1.1 Purpose
The Shadow BLE Protocol provides reliable, low-power communication for real-time stress monitoring data between embedded devices and mobile/desktop applications.

### 1.2 Key Features
- **Ring Buffer Management**: 32-entry circular buffer with 7-bit sequence numbering
- **State Synchronization**: Binary stress state tracking (CALM=0, STRESS=1) 
- **Data Loss Recovery**: Automatic detection and recovery of missed transitions
- **Power Efficiency**: Advertisement-based discovery with connection-on-demand
- **Reset Handling**: Protocol-level reset with sequence counter management

---

## 2. Protocol Architecture

### 2.1 Service Definition
- **Service UUID**: `A000`
- **Event Characteristic UUID**: `A002`
- **Device Name**: `"Shadow"`

### 2.2 Communication Model
```
ESP32 Device     <--> BLE Advertisement <--> macOS Client <--> Core Data
     |                        |                    |              |
 [Ring Buffer]    [State Broadcast]    [Synchronization]   [Persistence]
```

### 2.3 State Machine
```
IDLE → ADVERTISING → CONNECTED → REQUESTING → SYNCING → ADVERTISING
 ↓         ↓            ↓           ↓           ↓          ↓
ERROR ← CONNECTION_FAIL ← PROTOCOL_ERROR ← TIMEOUT ← RETRY
```

---

## 3. Packet Formats

### 3.1 Advertisement Packet
```
Byte 0: SSSSSSS S
        ||||||| |
        ||||||| +-- State bit (0=CALM, 1=STRESS)
        +---------- 7-bit sequence number (0-127)
```

**Example**: `0x0B` = sequence 5, stress state  
**Calculation**: `(5 << 1) | 1 = 0x0B`

### 3.2 Reset Request
```
Byte 0: 0xFF (Reset opcode)
```

### 3.3 Reset Response
```
Byte 0: 0x00 (Reserved)
Byte 1: XXXXXX S (Current state)
Byte 2: 0x00 (Reserved) 
Byte 3: 0x52 (Reset magic confirmation)
```

### 3.4 Missed Data Request
```
Byte 0: LLLLLLL (Last known sequence, 7-bit)
```

### 3.5 Minimal Response (Delta = 1)
```
Byte 0: CCCCCCC (Current sequence)
Byte 1: XXXXXX S (Current state)
```

### 3.6 Extended Response (Delta > 1)
```
Byte 0: CCCCCCC (Current sequence)
Byte 1: XXXXXX S (Current state)
Byte 2: MMMMMMM (Missed count)
Byte 3+: [Sequence, State] pairs for each missed entry
```

---

## 4. Protocol Operations

### 4.1 Device Discovery
1. ESP32 continuously advertises with service data
2. macOS scans for devices named "Shadow" with service A000
3. Advertisement data contains latest sequence and state

### 4.2 Synchronization Flow
```
macOS → ESP32: Connect to BLE device
macOS → ESP32: Discover service A000, characteristic A002  
macOS → ESP32: Write last known sequence number
ESP32 → macOS: Read response with current state + missed data
macOS → ESP32: Disconnect
```

### 4.3 Delta Calculation
```c
uint8_t delta = (new_sequence - last_sequence) & 0x7F;
```
- **Delta = 0**: No new data, ignore
- **Delta = 1**: Single transition, can use advertisement only (optional)
- **Delta > 1**: Multiple transitions, requires connection
- **Delta > 32**: Ring buffer overflow, requires reset

### 4.4 Reset Protocol
Used when data loss exceeds ring buffer capacity:
```
macOS → ESP32: Write 0xFF (reset opcode)
ESP32 → macOS: Read reset confirmation with current state
ESP32: Clear ring buffer, reset sequence to 0
macOS: Increment reset counter, persist reset marker
```

---

## 5. Error Handling

### 5.1 Connection Failures
- **Timeout**: Return to scanning mode
- **Service Discovery Fail**: Retry connection after throttle period (1.5s)
- **Characteristic Missing**: Log error, disconnect

### 5.2 Protocol Errors
- **Malformed Response**: Log error, request retry
- **Unexpected Data Length**: Log warning, attempt to parse
- **Reset Magic Mismatch**: Continue with normal parsing

### 5.3 Data Integrity
- **Sequence Gaps**: Automatic gap detection and recovery
- **State Validation**: Binary validation (0 or 1 only)
- **Duplicate Detection**: Check against existing database entries

---

## 6. Performance Characteristics

### 6.1 Timing Parameters
- **Advertisement Interval**: 1000ms typical
- **Connection Throttle**: 1.5s minimum between attempts
- **Ring Buffer Size**: 32 entries (supports ~32 seconds of transitions)
- **Sequence Wrap**: 128 values (7-bit), wraps every ~2 minutes

### 6.2 Power Consumption
- **Advertisement Mode**: ~50μA continuous
- **Connection Mode**: ~10mA for 100-500ms
- **Total Average**: <100μA with typical usage patterns

### 6.3 Data Throughput
- **Advertisement**: 1 byte per second
- **Sync Burst**: Up to 67 bytes (32 missed entries + headers)
- **Latency**: <2 seconds from state change to app notification

---

## 7. Security Considerations

### 7.1 Data Protection
- **Encryption**: Relies on BLE link-layer encryption
- **Authentication**: Device name verification only
- **Privacy**: No personally identifiable information in broadcasts

### 7.2 Denial of Service
- **Connection Throttling**: Prevents rapid connection attempts
- **Advertisement Validation**: Rejects malformed packets
- **Resource Limits**: Ring buffer prevents memory exhaustion

---

## 8. Implementation Notes

### 8.1 ESP32 Firmware
- **FreeRTOS Tasks**: Separate tasks for sensing, ML inference, and BLE
- **Ring Buffer**: Interrupt-safe circular buffer implementation
- **Power Management**: Sleep modes between advertisements

### 8.2 macOS Client
- **Core Bluetooth**: Native iOS/macOS BLE framework
- **Core Data**: SQLite-backed persistence with relationship management
- **SwiftUI**: Reactive UI updates from published BLE manager state

### 8.3 Cross-Platform Compatibility
- **Standard BLE**: Uses standard GATT services and characteristics
- **Endianness**: Little-endian byte order throughout
- **MTU Requirements**: Minimum 23 bytes (standard BLE minimum)

---

## 9. Future Extensions

### 9.1 Planned Features
- **Firmware Timestamp**: 64-bit device timestamps for events
- **Confidence Scores**: ML model confidence levels
- **Battery Monitoring**: Voltage level reporting
- **Sensor Quality**: Data quality metrics

### 9.2 Protocol Versioning
- **Version Field**: Reserved bit in advertisement packet
- **Capability Negotiation**: Service characteristic for feature discovery
- **Backward Compatibility**: Protocol designed for extensibility

---

## 10. References

- **Bluetooth Core Specification 5.0+**
- **ESP32-S3 Technical Reference Manual**
- **Apple Core Bluetooth Programming Guide**
- **Shadow Firmware Implementation (C/FreeRTOS)**
- **Shadow macOS Application (Swift/SwiftUI)**

---

**End of Document**

*This specification is implemented in the Shadow stress detection system as of December 2025.*
""")
        
        print(f"✅ RFC document generated: {rfc_path}")
    
    def save_all_visualizations(self):
        """Generate and save all BLE protocol visualizations"""
        print(f"🎨 Generating Shadow BLE protocol visualizations in {self.output_dir}")
        
        # 1. BLE Protocol Stack
        print("📊 Creating BLE protocol stack diagram...")
        fig1 = self.plot_ble_protocol_stack()
        fig1.savefig(self.output_dir / 'ble_protocol_stack.png', 
                    dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # 2. Data Transmission Pipeline
        print("📊 Creating data transmission pipeline...")
        fig2 = self.plot_data_transmission_pipeline()
        fig2.savefig(self.output_dir / 'data_transmission_pipeline.png', 
                    dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # 3. Message Flow Sequence
        print("📊 Creating message flow sequence...")
        fig3 = self.plot_message_flow_sequence()
        fig3.savefig(self.output_dir / 'message_flow_sequence.png', 
                    dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        # 4. Neural Network Architecture
        print("📊 Creating neural network architecture diagram...")
        fig4 = self.plot_neural_network_architecture()
        fig4.savefig(self.output_dir / 'neural_network_architecture.png', 
                    dpi=300, bbox_inches='tight')
        plt.close(fig4)
        
        # 5. Generate BLE RFC document
        print("📄 Generating BLE RFC document...")
        self.generate_ble_rfc_document()
        
        print(f"✅ All BLE protocol visualizations saved to: {self.output_dir}")
        return self.output_dir

def main():
    """Main execution function"""
    # Initialize visualizer
    visualizer = ShadowBLEProtocolVisualizer()
    
    # Generate all BLE protocol visualizations
    visualizer.save_all_visualizations()
    
    print("\n🎉 Shadow BLE protocol visualization complete!")
    print(f"📁 Check the visualizations: {visualizer.output_dir}")

if __name__ == "__main__":
    main()
