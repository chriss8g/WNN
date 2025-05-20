import torch

# Implementación de la función wavelet DB6 con dilatación y traslación
def db6_wavelet(x, a, b):
    # Coeficientes de la wavelet Daubechies 6 (DB6)
    c = torch.tensor([
        -0.001077301085308, 0.0047772575109455, 0.019787513133215,
        -0.061634441194945, -0.055754088222663, 0.32580342944987,
        0.751133908021095, 0.469762241707953, -0.143906003929106,
        -0.109702650336228, 0.028401554296343, 0.016602105764534,
        -0.004146564009503, -0.001629492012602, 0.000400216559711,
        0.000127069810248
    ], dtype=torch.float32).view(1, 1, -1)  # Forma requerida para conv1d
    
    # Aplicar dilatación y traslación
    x = (x - b) / a
    
    # Asegurarse de que x tenga la forma correcta para conv1d: (batch_size, channels, length)
    x = x.unsqueeze(1)  # Añadir dimensión de canales
    output = torch.nn.functional.conv1d(x, c, padding=c.shape[2] // 2)
    
    # Devolver el resultado sin la dimensión de canales
    output = output.squeeze(1)
    
    # Asegurarse de que la salida tenga exactamente 50 características
    if output.shape[1] > 50:
        output = output[:, :50]  # Recortar si hay más de 50 características
    elif output.shape[1] < 50:
        # Rellenar con ceros si hay menos de 50 características
        padding = torch.zeros(output.shape[0], 50 - output.shape[1])
        output = torch.cat([output, padding], dim=1)
    
    return output

