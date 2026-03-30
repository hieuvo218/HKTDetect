import math
import numpy as np
import torch


def infoNCE_image_text(v_s_pairs, temperature=0.07):
    """
    Compute the InfoNCE loss between image and text embeddings.

    Args:
        i_k (torch.Tensor): Text embeddings of shape (batch_size, embedding_dim).
        t_k (torch.Tensor): Image embeddings of shape (batch_size, embedding_dim).
        temperature (float): Temperature scaling factor.
    """
    v_k = v_k / np.linalg.norm(v_k)
    s_k = s_k / np.linalg.norm(s_k)

    sum_of_remains = 0
    loss_itot = 0
    for v_k, s_k in v_s_pairs:
        sum = 0
        for _, s_b in v_s_pairs:
            sum += np.exp(np.dot(v_k, s_b) / temperature)
        loss_itot += -np.log(np.exp(np.dot(v_k, s_k) / temperature) / sum)  # anchor is v_k
    
    loss_ttoi = 0 
    for v_k, s_k in v_s_pairs:
        sum = 0
        for v_b, _ in v_s_pairs:
            sum += np.exp(np.dot(s_k, v_b) / temperature)
        loss_ttoi += -np.log(np.exp(np.dot(s_k, v_k) / temperature) / sum)  # anchor is s_k

    return (loss_itot + loss_ttoi) / (2 * len(v_s_pairs))


def CRD_loss(teacher_pairs, student_pairs, temperature_t, temperature_s):
    """
    Docstring for CRD_loss
    
    :param teacher_pairs: visual and text embeddings from teacher
    :param student_pairs: visual and text embeddings from stuent
    """

    # with v_k as anchor
    for teacher_pair, student_pair in zip(teacher_pairs, student_pairs):
        vt_k, st_k = teacher_pair
        vs_k, ss_k = student_pair
        vt_k = vt_k / np.linalg.norm(vt_k)
        st_k = st_k / np.linalg.norm(st_k)
        vs_k = vs_k / np.linalg.norm(vs_k)
        ss_k = ss_k / np.linalg.norm(ss_k)

    p_t, p_s = [], []
    for k in range(len(teacher_pairs)):
        pt_k, ps_k = [], []
        vt_k, st_k = teacher_pairs[k]
        vs_k, ss_k = student_pairs[k]
        sum_s = 0, sum_t = 0
        
        for t_pair, s_pair in zip(teacher_pairs, student_pairs):
            _, st_b = t_pair
            _, ss_b = s_pair
            sum_s += np.exp(np.dot(vt_k, st_b) / temperature_t)
            sum_t += np.exp(np.dot(vs_k, ss_b) / temperature_t)

        for t_pair, s_pair in zip(teacher_pairs, student_pairs):
            _, st_j= t_pair
            _, ss_j = s_pair
            pt_k.append(np.exp(np.dot(vt_k, st_j / temperature_t)) / sum_t)
            ps_k.append(np.exp(np.dot(vs_k, ss_j / temperature_s)) / sum_s)

        p_t.append(pt_k)
        p_s.append(ps_k)
        
    q_t, q_s = [], []
    for k in range(len(teacher_pairs)):
        qt_k, qs_k = [], []
        vt_k, st_k = teacher_pairs[k]
        vs_k, ss_k = student_pairs[k]
        sum_s = 0, sum_t = 0
        
        for t_pair, s_pair in zip(teacher_pairs, student_pairs):
            vt_b, _ = t_pair
            vs_b, _ = s_pair
            sum_s += np.exp(np.dot(st_k, vt_b) / temperature_t)
            sum_t += np.exp(np.dot(ss_k, vs_b) / temperature_t)

        for t_pair, s_pair in zip(teacher_pairs, student_pairs):
            vt_j, _ = t_pair
            vs_j, _ = s_pair
            qt_k.append(np.exp(np.dot(st_k, vt_j / temperature_t)) / sum_t)
            qs_k.append(np.exp(np.dot(ss_k, vs_j / temperature_s)) / sum_s)

        q_t.append(qt_k)
        q_s.append(qs_k)

    loss_crd_itot, loss_crd_ttoi = 0, 0
    B = len(teacher_pairs)
    for k in range(B):
        for j in range(B):
            loss_crd_itot += p_t[k][j] * np.log(p_t[k][j] / p_s[k][j])
            loss_crd_ttoi += q_t[k][j] * np.log(q_t[k][j] / q_s[k][j])
    loss_crd_itot /= B
    loss_crd_ttoi /= B

    return loss_crd_ttoi + loss_crd_itot