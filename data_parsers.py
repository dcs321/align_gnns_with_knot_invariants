from sympy import symbols, sympify, I, pi, re, im, exp
import pandas as pd


def parse_volume(volume_str):
    return [[float(volume_str)]]

def parse_determinant(determinant_str):
    return [[float(determinant_str)]]

def parse_longitude_length(longitude_length_str):
    if pd.isna(longitude_length_str) or longitude_length_str.strip() == "Not Hyperbolic":
        return [[None]]
    return [[float(longitude_length_str)]]

def parse_meridian_length(meridian_length_str):
    if pd.isna(meridian_length_str) or meridian_length_str.strip() == "Not Hyperbolic":
        return [[None]]
    return [[float(meridian_length_str)]]

def parse_three_colorability(three_colorability_str):
    return [int(three_colorability_str)]

def parse_crossing_number(crossing_number_str):
    return [int(crossing_number_str)]

def parse_unknotting_number(unknotting_number_str):
    if pd.isna(unknotting_number_str) or unknotting_number_str.strip()[0] == "[":
        return [None]
    return [int(unknotting_number_str)]

def parse_genus_3d(genus_3d_str):
    return [int(genus_3d_str)]

def parse_signature(signature_str):
    return [int(signature_str)]

def parse_genus_4d(genus_4d_str):
    if pd.isna(genus_4d_str) or genus_4d_str.strip()[0] == "[":
        return [None]
    return [int(genus_4d_str)]

def parse_genus_4d_top(genus_4d_top_str):
    if pd.isna(genus_4d_top_str) or genus_4d_top_str.strip()[0] == "[":
        return [None]
    return [int(genus_4d_top_str)]

def parse_arf_invariant(arf_invariant_str):
    return [int(arf_invariant_str)]

def parse_rasmussen_s_invariant(rasmussen_s_invariant_str):
    return [int(rasmussen_s_invariant_str)]

def parse_ozsvath_szabo_tau_invariant(ozsvath_szabo_tau_invariant_str):
    return [int(ozsvath_szabo_tau_invariant_str)]

def parse_alternating(alternating_str):
    if alternating_str.strip() == "Y":
        value = 1
    elif alternating_str.strip() == "N":
        value = 0
    else:
        raise ValueError(f"Unexpected value for alternating: {alternating_str}")
    return [value]

def parse_jones_real_at_complex(jones_str):
    t = symbols('t')
    jones_str_simpy = jones_str.replace('^', '**')
    polynomial = sympify(jones_str_simpy)
    complex_value = exp(I * pi / 3)
    
    result_complex = polynomial.subs(t, complex_value)
    
    real_part = re(result_complex).evalf()


    return [int(real_part)]

def parse_jones_imaginary_at_complex(jones_str):
    t = symbols('t')
    jones_str_simpy = jones_str.replace('^', '**')
    polynomial = sympify(jones_str_simpy)
    complex_value = exp(I * pi / 3)
    
    result_complex = polynomial.subs(t, complex_value)
    
    imag_part = im(result_complex).evalf()

    return [int(imag_part)]

def parse_jones_real_and_imaginary_at_complex(jones_str):
    t = symbols('t')
    jones_str_simpy = jones_str.replace('^', '**')
    polynomial = sympify(jones_str_simpy)
    complex_value = exp(I * pi / 3)
    
    result_complex = polynomial.subs(t, complex_value)
    
    imag_part = im(result_complex).evalf()
    real_part = re(result_complex).evalf()
    return [int(real_part), int(imag_part)]

def parse_pd_notation(pd_str):
    pd_str = pd_str.replace(' ', '')
    outer_brackets_removed = pd_str[2:-2]
    splitted_by_lists = outer_brackets_removed.split('];[')
    pd_notation = []
    for sublist_str in splitted_by_lists:
        numbers_strs = sublist_str.split(';')
        sublist = []
        for num_str in numbers_strs:
            sublist.append(int(num_str))
        pd_notation.append(sublist)
    return pd_notation

def parse_list_of_features(list_of_feature_str, parse_function):
    parsed_features = []
    for feature_str in list_of_feature_str:
        parsed_feature = parse_function(feature_str)
        parsed_features.append(parsed_feature)
    return parsed_features