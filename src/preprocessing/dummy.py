import numpy as np

def angle_between_points(point1, point2, reference_point):
    vector1 = point1 - reference_point
    vector2 = point2 - reference_point

    dot_product = np.dot(vector1, vector2)
    magnitudes_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)

    cos_angle = dot_product / magnitudes_product
    angle_in_radians = np.arccos(cos_angle)
    angle_in_degrees = np.degrees(angle_in_radians)

    return angle_in_degrees


# Define the three points in 3D space
point1 = np.array([1, 0, 0])
point2 = np.array([0, 1, 0])
reference_point = np.array([0, 0, 0])

angle = angle_between_points(point1, point2, reference_point)
print("Angle between point1 and point2:", angle, "degrees")





