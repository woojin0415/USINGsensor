import numpy as np
from numpy.linalg import inv


A = np.array([[1, 0],
              [0, 1]])
H = np.array([[1, 0]])
Q = np.array([[1, 0],
              [0, 3]])
R = np.array([[40]])



def kalman_filter(z_meas, x_esti, P):
    """Kalman Filter Algorithm."""
    """칼만필터 알고리즘 (매개변수: 측정값, 추정값, 오차공분산)"""

    # (1) Prediction.
    """x와 P의 계산
    Input: 직전추정값과 오차공분산
    OutputL 예측값 """
    x_pred = A @ x_esti
    P_pred = A @ P @ A.T + Q

    # (2) Kalman Gain.
    """Kalman Gain 계산 
        P는 (1)에서 계산
        H와 R은 미리 설정된 값"""
    K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)

    # (3) Estimation.
    """입력된 측정값으로 추정값 계산
        X_pred는 (1)에서 계산"""
    x_esti = x_pred + K @ (z_meas - H @ x_pred)

    # (4) Error Covariance.
    """오차공분산 구하기
        앞에서 계산한 추정값을 사용할지 말지를 결정"""
    P = P_pred - K @ H @ P_pred

    #  추정값과 오차공분산
    return x_esti, P

def kamanfilter(input):
    x_esti, P = np.array([input[0],0]), 100 * np.eye(2)
    for i in range(len(input)):
        x_esti, P = kalman_filter(input[i], x_esti, P)
        input[i] = x_esti[0]

