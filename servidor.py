import tensorflow as tf
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
import os

# Carregar el model desat (cal haver-lo desat prèviament amb model.save('modelo_mnist.keras'))
if not os.path.exists('modelo_mnist.keras'):
    print("Error: El model no es troba. Primer entrena i desa el model.")
    exit(1)

# Carregar el model
model = tf.keras.models.load_model('modelo_mnist.keras')
print("Model carregat correctament")

# Noms de les classes en català
class_names = ['Zero', 'U', 'Dos', 'Tres', 'Quatre', 'Cinc', 'Sis', 'Set', 'Vuit', 'Nou']

class NeuralNetworkHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        # Gestionar CORS per a sol·licituds preflight
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
    def do_POST(self):
        # Configurar capçaleres per CORS
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        # Obtenir les dades del canvas
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        params = urllib.parse.parse_qs(post_data)
        
        # Convertir les dades del dibuix a un array de numpy
        pixels_str = params['pixeles'][0]
        pixels = np.array([float(p) for p in pixels_str.split(',')], dtype=np.float32)
        
        # Reformatejar els píxels a la forma esperada pel model (1, 28, 28, 1)
        pixels = pixels.reshape(1, 28, 28, 1)
        
        # Fer la predicció
        predictions = model.predict(pixels)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Retornar la predicció
        response = f"{class_names[predicted_class]} (Confiança: {confidence:.2f})"
        self.wfile.write(response.encode())

# Iniciar el servidor
def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, NeuralNetworkHandler)
    print(f"Servidor HTTP iniciat al port {port}...")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()