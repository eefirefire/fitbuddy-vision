import numpy as np

def get_angle(a, b, c):
    """Calculates angle between three points (a, b, c). b is the vertex."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_vertical_angle(p1, p2):
    """
    Calculates the angle of a body segment (like the torso) relative to vertical.
    0 degrees = perfectly upright.
    """
    a = np.array(p1) # Shoulder
    b = np.array(p2) # Hip
    
    # Create a virtual vertical point above the hip
    c = np.array([b[0], b[1] - 100]) 
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def check_valgus(hip, knee, ankle):
    """
    Checks if the knee is caving inward (Valgus).
    Returns the horizontal distance the knee has collapsed inward.
    Positive value = Knee is INSIDE the Hip-Ankle line (Bad).
    """
    # Simply check the X-coordinate difference relative to the ankle/hip alignment
    # If Knee X is significantly 'inside' the Ankle X (towards the other leg), it's Valgus.
    # Note: This is a simplified 2D approximation for the MVP.
    
    # Calculate the x-coordinate of the point on the Hip-Ankle line at the Knee's height
    # (Simplified for front/45-degree view)
    expected_knee_x = (hip[0] + ankle[0]) / 2
    
    # Deviation: How far is the actual knee from the center line?
    # We return the difference based on logic in the main loop
    return knee[0] - expected_knee_x