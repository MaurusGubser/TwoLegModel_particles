import numpy as np

g = 9.81
DIM_STATES = 18
DIM_OBSERVATIONS = 20

def state_to_obs(x, legs, imus):
    """
    Compute the observation vector for a given collection of state vectors
    :param x: np.ndarray
        N x 18-dimensional state vector where N is the number of particles and the 18
        dimension composed of $ (x_H, y_H, phi_0, phi_1, phi_2, phi_3) $ and the
        corresponding first and second (temporal) derivatives
    :param legs: np.ndarray
            4-dimensional array, containing the length of left femur, left fibula,
            right femur, right fibula
    :param imus: np.ndarray
            4-dimensional array, containing the position of the four imus, measured from
            the hip or the knees, respectively
    :return: np.ndarray
        N x dim_observation-dimensional observation vectors
    """
    nb_steps, _ = x.shape
    y = np.empty((nb_steps, DIM_OBSERVATIONS))
    # left femur
    y[:, 0] = imus[0] * x[:, 14] + g * np.sin(x[:, 2]) + np.sin(x[:, 2]) * x[:, 13] + np.cos(x[:, 2]) * x[:, 12]
    y[:, 1] = imus[0] * x[:, 8] ** 2 + g * np.cos(x[:, 2]) - np.sin(x[:, 2]) * x[:, 12] + np.cos(x[:, 2]) * x[:, 13]
    y[:, 2] = x[:, 8]

    # left fibula
    y[:, 3] = (
        imus[1] * x[:, 14]
        + imus[1] * x[:, 15]
        + g * np.sin(x[:, 2] + x[:, 3])
        + legs[0] * np.sin(x[:, 3]) * x[:, 8] ** 2
        + legs[0] * np.cos(x[:, 3]) * x[:, 14]
        + np.sin(x[:, 2] + x[:, 3]) * x[:, 13]
        + np.cos(x[:, 2] + x[:, 3]) * x[:, 12]
    )
    y[:, 4] = (
        imus[1] * x[:, 8] ** 2
        + 2 * imus[1] * x[:, 8] * x[:, 9]
        + imus[1] * x[:, 9] ** 2
        + g * np.cos(x[:, 2] + x[:, 3])
        - legs[0] * np.sin(x[:, 3]) * x[:, 14]
        + legs[0] * np.cos(x[:, 3]) * x[:, 8] ** 2
        - np.sin(x[:, 2] + x[:, 3]) * x[:, 12]
        + np.cos(x[:, 2] + x[:, 3]) * x[:, 13]
    )
    y[:, 5] = x[:, 8] + x[:, 9]

    # right femur
    y[:, 6] = imus[2] * x[:, 16] + g * np.sin(x[:, 4]) + np.sin(x[:, 4]) * x[:, 13] + np.cos(x[:, 4]) * x[:, 12]
    y[:, 7] = imus[2] * x[:, 10] ** 2 + g * np.cos(x[:, 4]) - np.sin(x[:, 4]) * x[:, 12] + np.cos(x[:, 4]) * x[:, 13]
    y[:, 8] = x[:, 10]

    # right fibula
    y[:, 9] = (
        imus[3] * x[:, 16]
        + imus[3] * x[:, 17]
        + g * np.sin(x[:, 4] + x[:, 5])
        + legs[2] * np.sin(x[:, 5]) * x[:, 10] ** 2
        + legs[2] * np.cos(x[:, 5]) * x[:, 16]
        + np.sin(x[:, 4] + x[:, 5]) * x[:, 13]
        + np.cos(x[:, 4] + x[:, 5]) * x[:, 12]
    )
    y[:, 10] = (
        imus[3] * x[:, 10] ** 2
        + 2 * imus[3] * x[:, 10] * x[:, 11]
        + imus[3] * x[:, 11] ** 2
        + g * np.cos(x[:, 4] + x[:, 5])
        - legs[2] * np.sin(x[:, 5]) * x[:, 16]
        + legs[2] * np.cos(x[:, 5]) * x[:, 10] ** 2
        - np.sin(x[:, 4] + x[:, 5]) * x[:, 12]
        + np.cos(x[:, 4] + x[:, 5]) * x[:, 13]
    )
    y[:, 11] = x[:, 10] + x[:, 11]

    # left heel
    y[:, 12] = (
        legs[0] * np.cos(x[:, 2]) * x[:, 8] + legs[1] * (x[:, 8] + x[:, 9]) * np.cos(x[:, 2] + x[:, 3]) + x[:, 6]
    )
    y[:, 13] = (
        legs[0] * np.sin(x[:, 2]) * x[:, 8] + legs[1] * (x[:, 8] + x[:, 9]) * np.sin(x[:, 2] + x[:, 3]) + x[:, 7]
    )
    y[:, 14] = (
        -legs[0] * np.sin(x[:, 2]) * x[:, 8] ** 2
        + legs[0] * np.cos(x[:, 2]) * x[:, 14]
        - legs[1] * (x[:, 8] + x[:, 9]) ** 2 * np.sin(x[:, 2] + x[:, 3])
        + legs[1] * (x[:, 14] + x[:, 15]) * np.cos(x[:, 2] + x[:, 3])
        + x[:, 12]
    )
    y[:, 15] = (
        legs[0] * np.sin(x[:, 2]) * x[:, 14]
        + legs[0] * np.cos(x[:, 2]) * x[:, 8] ** 2
        + legs[1] * (x[:, 8] + x[:, 9]) ** 2 * np.cos(x[:, 2] + x[:, 3])
        + legs[1] * (x[:, 14] + x[:, 15]) * np.sin(x[:, 2] + x[:, 3])
        + x[:, 13]
    )

    # right heel
    y[:, 16] = (
        legs[2] * np.cos(x[:, 4]) * x[:, 10] + legs[3] * (x[:, 10] + x[:, 11]) * np.cos(x[:, 4] + x[:, 5]) + x[:, 6]
    )
    y[:, 17] = (
        legs[2] * np.sin(x[:, 4]) * x[:, 10] + legs[3] * (x[:, 10] + x[:, 11]) * np.sin(x[:, 4] + x[:, 5]) + x[:, 7]
    )
    y[:, 18] = (
        -legs[2] * np.sin(x[:, 4]) * x[:, 10] ** 2
        + legs[2] * np.cos(x[:, 4]) * x[:, 16]
        - legs[3] * (x[:, 10] + x[:, 11]) ** 2 * np.sin(x[:, 4] + x[:, 5])
        + legs[3] * (x[:, 16] + x[:, 17]) * np.cos(x[:, 4] + x[:, 5])
        + x[:, 12]
    )
    y[:, 19] = (
        legs[2] * np.sin(x[:, 4]) * x[:, 16]
        + legs[2] * np.cos(x[:, 4]) * x[:, 10] ** 2
        + legs[3] * (x[:, 10] + x[:, 11]) ** 2 * np.cos(x[:, 4] + x[:, 5])
        + legs[3] * (x[:, 16] + x[:, 17]) * np.sin(x[:, 4] + x[:, 5])
        + x[:, 13]
    )

    return y
    
    
def compute_jacobian_obs(x, legs, imus):
    """
    Compute the Jacobian of the state-observation-transition at a given point
    :param x: np.ndarray
        N x 18-dimensional state vector where N is the number of particles and the 18
        dimension composed of $ (x_H, y_H, phi_0, phi_1, phi_2, phi_3) $ and the
        corresponding first and second (temporal) derivatives
    :param legs: np.ndarray
            4-dimensional array, containing the length of left femur, left fibula,
            right femur, right fibula
    :param imus: np.ndarray
            4-dimensional array, containing the position of the four imus, measured from
            the hip or the knees, respectively
    :return: np.ndarray
        N x dim_states x dim_observation-dimensional Jacobian
    """
    nb_particles, _ = x.shape
    df = np.zeros((nb_particles, 20, DIM_STATES))

    # left femur
    df[:, 0, 2] = -x[:, 12] * np.sin(x[:, 2]) + (x[:, 13] + g) * np.cos(x[:, 2])
    df[:, 0, 12] = np.cos(x[:, 2])
    df[:, 0, 13] = np.sin(x[:, 2])
    df[:, 0, 14] = imus[0]
    df[:, 1, 2] = -x[:, 12] * np.cos(x[:, 2]) - (x[:, 13] + g) * np.sin(x[:, 2])
    df[:, 1, 8] = 2 * x[:, 8] * imus[0]
    df[:, 1, 12] = -np.sin(x[:, 2])
    df[:, 1, 13] = np.cos(x[:, 2])
    df[:, 2, 8] = 1

    # left fibula
    df[:, 3, 2] = (
        -x[:, 12] * np.sin(x[:, 2] + x[:, 3]) + x[:, 13] * np.cos(x[:, 2] + x[:, 3]) + g * np.cos(x[:, 2] + x[:, 3])
    )
    df[:, 3, 3] = (
        -x[:, 14] * legs[0] * np.sin(x[:, 3])
        - x[:, 12] * np.sin(x[:, 2] + x[:, 3])
        + x[:, 13] * np.cos(x[:, 2] + x[:, 3])
        + legs[0] * x[:, 8] ** 2 * np.cos(x[:, 3])
        + g * np.cos(x[:, 2] + x[:, 3])
    )
    df[:, 3, 8] = 2 * legs[0] * x[:, 8] * np.sin(x[:, 3])
    df[:, 3, 12] = np.cos(x[:, 2] + x[:, 3])
    df[:, 3, 13] = np.sin(x[:, 2] + x[:, 3])
    df[:, 3, 14] = imus[1] + legs[0] * np.cos(x[:, 3])
    df[:, 3, 15] = imus[1]
    df[:, 4, 2] = (
        -x[:, 12] * np.cos(x[:, 2] + x[:, 3]) - x[:, 13] * np.sin(x[:, 2] + x[:, 3]) - g * np.sin(x[:, 2] + x[:, 3])
    )
    df[:, 4, 3] = (
        -x[:, 14] * legs[0] * np.cos(x[:, 3])
        - x[:, 12] * np.cos(x[:, 2] + x[:, 3])
        - x[:, 13] * np.sin(x[:, 2] + x[:, 3])
        - legs[0] * x[:, 8] ** 2 * np.sin(x[:, 3])
        - g * np.sin(x[:, 2] + x[:, 3])
    )
    df[:, 4, 8] = 2 * legs[0] * x[:, 8] * np.cos(x[:, 3]) + 2 * imus[1] * (x[:, 8] + x[:, 9])
    df[:, 4, 9] = 2 * imus[1] * (x[:, 8] + x[:, 9])
    df[:, 4, 12] = -np.sin(x[:, 2] + x[:, 3])
    df[:, 4, 13] = np.cos(x[:, 2] + x[:, 3])
    df[:, 4, 14] = -legs[0] * np.sin(x[:, 3])
    df[:, 5, 8] = 1
    df[:, 5, 9] = 1

    # right femur
    df[:, 6, 4] = -x[:, 12] * np.sin(x[:, 4]) + (x[:, 13] + g) * np.cos(x[:, 4])
    df[:, 6, 12] = np.cos(x[:, 4])
    df[:, 6, 13] = np.sin(x[:, 4])
    df[:, 6, 16] = imus[2]
    df[:, 7, 4] = -x[:, 12] * np.cos(x[:, 4]) - (x[:, 13] + g) * np.sin(x[:, 4])
    df[:, 7, 10] = 2 * x[:, 10] * imus[2]
    df[:, 7, 12] = -np.sin(x[:, 4])
    df[:, 7, 13] = np.cos(x[:, 4])
    df[:, 8, 10] = 1

    # right fibula
    df[:, 9, 4] = (
        -x[:, 12] * np.sin(x[:, 4] + x[:, 5]) + x[:, 13] * np.cos(x[:, 4] + x[:, 5]) + g * np.cos(x[:, 4] + x[:, 5])
    )
    df[:, 9, 5] = (
        -x[:, 16] * legs[2] * np.sin(x[:, 5])
        - x[:, 12] * np.sin(x[:, 4] + x[:, 5])
        + x[:, 13] * np.cos(x[:, 4] + x[:, 5])
        + legs[2] * x[:, 10] ** 2 * np.cos(x[:, 5])
        + g * np.cos(x[:, 4] + x[:, 5])
    )
    df[:, 9, 10] = 2 * legs[2] * x[:, 10] * np.sin(x[:, 5])
    df[:, 9, 12] = np.cos(x[:, 4] + x[:, 5])
    df[:, 9, 13] = np.sin(x[:, 4] + x[:, 5])
    df[:, 9, 16] = imus[3] + legs[2] * np.cos(x[:, 5])
    df[:, 9, 17] = imus[3]
    df[:, 10, 4] = (
        -x[:, 12] * np.cos(x[:, 4] + x[:, 5]) - x[:, 13] * np.sin(x[:, 4] + x[:, 5]) - g * np.sin(x[:, 4] + x[:, 5])
    )
    df[:, 10, 5] = (
        -x[:, 16] * legs[2] * np.cos(x[:, 5])
        - x[:, 12] * np.cos(x[:, 4] + x[:, 5])
        - x[:, 13] * np.sin(x[:, 4] + x[:, 5])
        - legs[2] * x[:, 10] ** 2 * np.sin(x[:, 5])
        - g * np.sin(x[:, 4] + x[:, 5])
    )
    df[:, 10, 10] = 2 * legs[2] * x[:, 10] * np.cos(x[:, 5]) + 2 * imus[3] * (x[:, 10] + x[:, 11])
    df[:, 10, 11] = 2 * imus[3] * (x[:, 10] + x[:, 11])
    df[:, 10, 12] = -np.sin(x[:, 4] + x[:, 5])
    df[:, 10, 13] = np.cos(x[:, 4] + x[:, 5])
    df[:, 10, 16] = -legs[2] * np.sin(x[:, 5])
    df[:, 11, 10] = 1
    df[:, 11, 11] = 1

    # left heel
    df[:, 12, 2] = -x[:, 8] * legs[0] + np.sin(x[:, 2]) - legs[1] * (x[:, 8] + x[:, 9]) * np.sin(x[:, 2] + x[:, 3])
    df[:, 12, 3] = -legs[1] * (x[:, 8] + x[:, 9]) * np.sin(x[:, 2] + x[:, 3])
    df[:, 12, 6] = 1
    df[:, 12, 8] = legs[0] * np.cos(x[:, 2]) + legs[1] * np.cos(x[:, 2] + x[:, 3])
    df[:, 12, 9] = legs[1] * np.cos(x[:, 2] + x[:, 3])
    df[:, 13, 2] = x[:, 8] * legs[0] * np.cos(x[:, 2]) + legs[1] * (x[:, 8] + x[:, 9]) * np.cos(x[:, 2] + x[:, 3])
    df[:, 13, 3] = legs[1] * (x[:, 8] + x[:, 9]) * np.cos(x[:, 2] + x[:, 3])
    df[:, 13, 7] = 1
    df[:, 13, 8] = legs[0] * np.sin(x[:, 2]) + legs[1] * np.sin(x[:, 2] + x[:, 3])
    df[:, 13, 9] = legs[1] * np.sin(x[:, 2] + x[:, 3])
    df[:, 14, 2] = (
        -legs[0] * (x[:, 14] * np.sin(x[:, 2]) + x[:, 8] ** 2 * np.cos(x[:, 2]))
        - legs[1] * (x[:, 14] + x[:, 15]) * np.sin(x[:, 2] + x[:, 3])
        - legs[1] * (x[:, 8] + x[:, 9]) ** 2 * np.cos(x[:, 2] + x[:, 3])
    )
    df[:, 14, 3] = -legs[1] * (x[:, 14] + x[:, 15]) * np.sin(x[:, 2] + x[:, 3]) - legs[1] * (
        x[:, 8] + x[:, 9]
    ) ** 2 * np.cos(x[:, 2] + x[:, 3])
    df[:, 14, 8] = -2 * legs[0] * x[:, 8] * np.sin(x[:, 2]) - 2 * legs[1] * (x[:, 8] + x[:, 9]) * np.sin(
        x[:, 2] + x[:, 3]
    )
    df[:, 14, 9] = -2 * legs[1] * (x[:, 8] + x[:, 9]) * np.sin(x[:, 2] + x[:, 3])
    df[:, 14, 12] = 1
    df[:, 14, 14] = legs[0] * np.cos(x[:, 2]) + legs[1] * np.cos(x[:, 2] + x[:, 3])
    df[:, 14, 15] = legs[1] * np.cos(x[:, 2] + x[:, 3])
    df[:, 15, 2] = (
        -legs[0] * (-x[:, 14] * np.cos(x[:, 2]) + x[:, 8] ** 2 * np.sin(x[:, 2]))
        + legs[1] * (x[:, 14] + x[:, 15]) * np.cos(x[:, 2] + x[:, 3])
        - legs[1] * (x[:, 8] + x[:, 9]) ** 2 * np.sin(x[:, 2] + x[:, 3])
    )
    df[:, 15, 3] = legs[1] * (x[:, 14] + x[:, 15]) * np.cos(x[:, 2] + x[:, 3]) - legs[1] * (
        x[:, 8] + x[:, 9]
    ) ** 2 * np.sin(x[:, 2] + x[:, 3])
    df[:, 15, 8] = 2 * legs[0] * x[:, 8] * np.cos(x[:, 2]) + 2 * legs[1] * (x[:, 8] + x[:, 9]) * np.cos(
        x[:, 2] + x[:, 3]
    )
    df[:, 15, 9] = 2 * legs[1] * (x[:, 8] + x[:, 9]) * np.cos(x[:, 2] + x[:, 3])
    df[:, 15, 13] = 1
    df[:, 15, 14] = legs[0] * np.sin(x[:, 2]) + legs[1] * np.sin(x[:, 2] + x[:, 3])
    df[:, 15, 15] = legs[1] * np.sin(x[:, 2] + x[:, 3])

    # right heel
    df[:, 16, 4] = -x[:, 10] * legs[2] + np.sin(x[:, 4]) - legs[3] * (x[:, 10] + x[:, 11]) * np.sin(x[:, 4] + x[:, 5])
    df[:, 16, 5] = -legs[3] * (x[:, 10] + x[:, 11]) * np.sin(x[:, 4] + x[:, 5])
    df[:, 16, 6] = 1
    df[:, 16, 10] = legs[2] * np.cos(x[:, 4]) + legs[3] * np.cos(x[:, 4] + x[:, 5])
    df[:, 16, 11] = legs[3] * np.cos(x[:, 4] + x[:, 5])
    df[:, 17, 4] = x[:, 10] * legs[2] * np.cos(x[:, 4]) + legs[3] * (x[:, 10] + x[:, 11]) * np.cos(x[:, 4] + x[:, 5])
    df[:, 17, 5] = legs[3] * (x[:, 10] + x[:, 11]) * np.cos(x[:, 4] + x[:, 5])
    df[:, 17, 7] = 1
    df[:, 17, 10] = legs[2] * np.sin(x[:, 4]) + legs[3] * np.sin(x[:, 4] + x[:, 5])
    df[:, 17, 11] = legs[3] * np.sin(x[:, 4] + x[:, 5])
    df[:, 18, 4] = (
        -legs[2] * (x[:, 16] * np.sin(x[:, 4]) + x[:, 10] ** 2 * np.cos(x[:, 4]))
        - legs[3] * (x[:, 16] + x[:, 17]) * np.sin(x[:, 4] + x[:, 5])
        - legs[3] * (x[:, 10] + x[:, 11]) ** 2 * np.cos(x[:, 4] + x[:, 5])
    )
    df[:, 18, 5] = -legs[3] * (x[:, 16] + x[:, 17]) * np.sin(x[:, 4] + x[:, 5]) - legs[3] * (
        x[:, 10] + x[:, 11]
    ) ** 2 * np.cos(x[:, 4] + x[:, 5])
    df[:, 18, 10] = -2 * legs[2] * x[:, 10] * np.sin(x[:, 4]) - 2 * legs[3] * (x[:, 10] + x[:, 11]) * np.sin(
        x[:, 4] + x[:, 5]
    )
    df[:, 18, 11] = -2 * legs[3] * (x[:, 10] + x[:, 11]) * np.sin(x[:, 4] + x[:, 5])
    df[:, 18, 12] = 1
    df[:, 18, 16] = legs[2] * np.cos(x[:, 4]) + legs[3] * np.cos(x[:, 4] + x[:, 5])
    df[:, 18, 17] = legs[3] * np.cos(x[:, 4] + x[:, 5])
    df[:, 19, 4] = (
        -legs[2] * (-x[:, 16] * np.cos(x[:, 4]) + x[:, 10] ** 2 * np.sin(x[:, 4]))
        + legs[3] * (x[:, 16] + x[:, 17]) * np.cos(x[:, 4] + x[:, 5])
        - legs[3] * (x[:, 10] + x[:, 11]) ** 2 * np.sin(x[:, 4] + x[:, 5])
    )
    df[:, 19, 5] = legs[3] * (x[:, 16] + x[:, 17]) * np.cos(x[:, 4] + x[:, 5]) - legs[3] * (
        x[:, 10] + x[:, 11]
    ) ** 2 * np.sin(x[:, 4] + x[:, 5])
    df[:, 19, 10] = 2 * legs[2] * x[:, 10] * np.cos(x[:, 4]) + 2 * legs[3] * (x[:, 10] + x[:, 11]) * np.cos(
        x[:, 4] + x[:, 5]
    )
    df[:, 19, 11] = 2 * legs[3] * (x[:, 10] + x[:, 11]) * np.cos(x[:, 4] + x[:, 5])
    df[:, 19, 13] = 1
    df[:, 19, 16] = legs[2] * np.sin(x[:, 4]) + legs[3] * np.sin(x[:, 4] + x[:, 5])
    df[:, 19, 17] = legs[3] * np.sin(x[:, 4] + x[:, 5])

    return df
