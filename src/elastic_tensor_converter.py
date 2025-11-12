# -*- coding: utf-8 -*-
"""
Elastic Tensor Converter
Convert elastic stiffness tensor (from MD/DFT) to engineering constants (for Abaqus/FE)

@author: nk03

This script converts a 6x6 stiffness matrix C (in Voigt notation) to engineering constants:
E1, E2, E3, nu12, nu13, nu23, G12, G13, G23

Used for: MD simulation results, DFT calculations, experimental measurements
"""

import numpy as np


def stiffness_to_compliance(C):
    """
    Convert stiffness matrix C to compliance matrix S
    
    Parameters:
    -----------
    C : np.ndarray (6x6)
        Stiffness matrix in Voigt notation (GPa)
        
    Returns:
    --------
    S : np.ndarray (6x6)
        Compliance matrix in Voigt notation (1/GPa)
    """
    S = np.linalg.inv(C)
    return S


def compliance_to_engineering_constants(S):
    """
    Extract engineering constants from compliance matrix
    
    Parameters:
    -----------
    S : np.ndarray (6x6)
        Compliance matrix (1/GPa)
        
    Returns:
    --------
    constants : dict
        Dictionary containing:
        - E1, E2, E3: Young's moduli (GPa)
        - nu12, nu13, nu23: Poisson's ratios
        - nu21, nu31, nu32: Reciprocal Poisson's ratios (for validation)
        - G12, G13, G23: Shear moduli (GPa)
    """
    # Young's moduli
    E1 = 1.0 / S[0, 0]
    E2 = 1.0 / S[1, 1]
    E3 = 1.0 / S[2, 2]
    
    # Poisson's ratios (nu_ij = -S_ij * E_i)
    nu12 = -S[0, 1] * E1
    nu13 = -S[0, 2] * E1
    nu23 = -S[1, 2] * E2
    
    # Reciprocal Poisson's ratios (for validation)
    nu21 = -S[1, 0] * E2
    nu31 = -S[2, 0] * E3
    nu32 = -S[2, 1] * E3
    
    # Shear moduli
    G23 = 1.0 / S[3, 3]  # S44 in Voigt notation
    G13 = 1.0 / S[4, 4]  # S55 in Voigt notation
    G12 = 1.0 / S[5, 5]  # S66 in Voigt notation
    
    constants = {
        'E1': E1, 'E2': E2, 'E3': E3,
        'nu12': nu12, 'nu13': nu13, 'nu23': nu23,
        'nu21': nu21, 'nu31': nu31, 'nu32': nu32,
        'G12': G12, 'G13': G13, 'G23': G23
    }
    
    return constants


def validate_orthotropic_symmetry(constants, tolerance=1e-3):
    """
    Validate orthotropic symmetry relations: nu_ij/E_i = nu_ji/E_j
    
    Parameters:
    -----------
    constants : dict
        Engineering constants from compliance_to_engineering_constants
    tolerance : float
        Relative tolerance for validation
        
    Returns:
    --------
    valid : bool
        True if all symmetry relations are satisfied
    errors : dict
        Relative errors for each relation
    """
    E1, E2, E3 = constants['E1'], constants['E2'], constants['E3']
    nu12, nu13, nu23 = constants['nu12'], constants['nu13'], constants['nu23']
    nu21, nu31, nu32 = constants['nu21'], constants['nu31'], constants['nu32']
    
    # Check symmetry relations
    error12 = abs((nu12/E1) - (nu21/E2)) / abs(nu12/E1)
    error13 = abs((nu13/E1) - (nu31/E3)) / abs(nu13/E1)
    error23 = abs((nu23/E2) - (nu32/E3)) / abs(nu23/E2)
    
    errors = {
        'nu12_nu21': error12,
        'nu13_nu31': error13,
        'nu23_nu32': error23
    }
    
    valid = all(err < tolerance for err in errors.values())
    
    return valid, errors


def format_for_abaqus(constants, unit='MPa'):
    """
    Format engineering constants for Abaqus *Elastic, type=ORTHOTROPIC
    
    Parameters:
    -----------
    constants : dict
        Engineering constants
    unit : str
        'GPa' or 'MPa'
        
    Returns:
    --------
    abaqus_line : str
        Formatted string for Abaqus input
    """
    E1, E2, E3 = constants['E1'], constants['E2'], constants['E3']
    nu12, nu13, nu23 = constants['nu12'], constants['nu13'], constants['nu23']
    G12, G13, G23 = constants['G12'], constants['G13'], constants['G23']
    
    # Convert to MPa if needed
    if unit.upper() == 'MPA':
        E1, E2, E3 = E1*1000, E2*1000, E3*1000
        G12, G13, G23 = G12*1000, G13*1000, G23*1000
    
    # Abaqus format: E1, E2, E3, nu12, nu13, nu23, G12, G13, G23
    abaqus_line = f"{E1:.6e}, {E2:.6e}, {E3:.6e}, {nu12:.6f}, {nu13:.6f}, {nu23:.6f}, {G12:.6e}, {G13:.6e}, {G23:.6e}"
    
    return abaqus_line


def print_summary(C, constants, valid, errors):
    """
    Print a nice summary of the conversion
    """
    print("="*70)
    print("ELASTIC TENSOR CONVERSION SUMMARY")
    print("="*70)
    print("\nInput Stiffness Matrix C (GPa):")
    print(C)
    print(f"\nDeterminant: {np.linalg.det(C):.6e} (should be > 0)")
    
    print("\n" + "-"*70)
    print("ENGINEERING CONSTANTS")
    print("-"*70)
    print("\nYoung's Moduli (GPa):")
    print(f"  E1 = {constants['E1']:.3f}")
    print(f"  E2 = {constants['E2']:.3f}")
    print(f"  E3 = {constants['E3']:.3f}")
    
    print("\nPoisson's Ratios:")
    print(f"  ν12 = {constants['nu12']:.4f}")
    print(f"  ν13 = {constants['nu13']:.4f}")
    print(f"  ν23 = {constants['nu23']:.4f}")
    
    print("\nShear Moduli (GPa):")
    print(f"  G12 = {constants['G12']:.3f}")
    print(f"  G13 = {constants['G13']:.3f}")
    print(f"  G23 = {constants['G23']:.3f}")
    
    print("\n" + "-"*70)
    print("VALIDATION")
    print("-"*70)
    print(f"\nOrthotropic symmetry valid: {valid}")
    print("\nSymmetry relation errors:")
    print(f"  ν12/E1 = ν21/E2: {errors['nu12_nu21']:.2e} (should be < 1e-3)")
    print(f"  ν13/E1 = ν31/E3: {errors['nu13_nu31']:.2e}")
    print(f"  ν23/E2 = ν32/E3: {errors['nu23_nu32']:.2e}")
    
    if not valid:
        print("\n⚠️  WARNING: Symmetry validation failed!")
        print("   Check your stiffness matrix or increase tolerance")
    
    print("\n" + "-"*70)
    print("FOR ABAQUS")
    print("-"*70)
    print("\n*Material, name=YourMaterial")
    print("*Elastic, type=ORTHOTROPIC")
    print("** E1, E2, E3, nu12, nu13, nu23, G12, G13, G23")
    print(format_for_abaqus(constants, unit='MPa'))
    
    print("\n" + "="*70)


def aragonite_example():
    """
    Example: Aragonite CaCO3 (orthorhombic)
    From DFT calculations or experiments
    """
    print("\n" + "="*70)
    print("EXAMPLE: ARAGONITE CaCO3")
    print("="*70)
    
    # Stiffness matrix from DFT (Dandekar & Ruoff, 1968)
    # Values in GPa
    C_aragonite = np.array([
        [171.8,  57.5,  30.2,   0.0,   0.0,   0.0],  # C11, C12, C13, C14, C15, C16
        [ 57.5,  106.7,  46.9,   0.0,   0.0,   0.0],  # C21, C22, C23, C24, C25, C26
        [ 30.2,  46.9,  84.2,   0.0,   0.0,   0.0],  # C31, C32, C33, C34, C35, C36
        [  0.0,   0.0,   0.0,  42.1,   0.0,   0.0],  # C41, C42, C43, C44, C45, C46
        [  0.0,   0.0,   0.0,   0.0,  31.1,   0.0],  # C51, C52, C53, C54, C55, C56
        [  0.0,   0.0,   0.0,   0.0,   0.0,  46.6]   # C61, C62, C63, C64, C65, C66
    ])
            
    # Convert to compliance
    S = stiffness_to_compliance(C_aragonite)
    
    # Extract engineering constants
    constants = compliance_to_engineering_constants(S)
    
    # Validate
    valid, errors = validate_orthotropic_symmetry(constants)
    
    # Print summary
    print_summary(C_aragonite, constants, valid, errors)
    
    return C_aragonite, constants


def bcc_iron_example():
    """
    Example: BCC Iron (cubic symmetry)
    For cubic: C11=C22=C33, C12=C13=C23, C44=C55=C66
    """
    print("\n" + "="*70)
    print("EXAMPLE: BCC IRON (CUBIC)")
    print("="*70)
    
    # Stiffness matrix for BCC Fe at 300K (from experiments)
    # Values in GPa
    C11 = 233.0
    C12 = 135.0
    C44 = 118.0
    
    C_bcc_fe = np.array([
        [C11, C12, C12,  0.0,  0.0,  0.0],
        [C12, C11, C12,  0.0,  0.0,  0.0],
        [C12, C12, C11,  0.0,  0.0,  0.0],
        [0.0, 0.0, 0.0,  C44,  0.0,  0.0],
        [0.0, 0.0, 0.0,  0.0,  C44,  0.0],
        [0.0, 0.0, 0.0,  0.0,  0.0,  C44]
    ])
    
    # Convert
    S = stiffness_to_compliance(C_bcc_fe)
    constants = compliance_to_engineering_constants(S)
    valid, errors = validate_orthotropic_symmetry(constants)
    
    # Print summary
    print_summary(C_bcc_fe, constants, valid, errors)
    
    # For cubic, can also use isotropic approximation
    print("\n" + "-"*70)
    print("ISOTROPIC APPROXIMATION (for cubic symmetry)")
    print("-"*70)
    # Voigt average
    K_voigt = (C11 + 2*C12) / 3  # Bulk modulus
    G_voigt = (C11 - C12 + 3*C44) / 5  # Shear modulus
    E_voigt = 9*K_voigt*G_voigt / (3*K_voigt + G_voigt)
    nu_voigt = (3*K_voigt - 2*G_voigt) / (2*(3*K_voigt + G_voigt))
    
    print(f"\nVoigt average:")
    print(f"  E = {E_voigt:.3f} GPa")
    print(f"  ν = {nu_voigt:.4f}")
    print(f"  G = {G_voigt:.3f} GPa")
    print(f"  K = {K_voigt:.3f} GPa")
    
    print("\nFor Abaqus (isotropic):")
    print("*Elastic")
    print(f"{E_voigt*1000:.6e}, {nu_voigt:.6f}")
    
    return C_bcc_fe, constants


def custom_tensor_input():
    """
    Interactive input for custom elastic tensor
    """
    print("\n" + "="*70)
    print("CUSTOM ELASTIC TENSOR INPUT")
    print("="*70)
    print("\nEnter your 6x6 stiffness matrix in GPa")
    print("(For symmetric tensors, only upper triangle is needed)")
    print("\nExample format:")
    print("C11 C12 C13 C14 C15 C16")
    print("    C22 C23 C24 C25 C26")
    print("        C33 C34 C35 C36")
    print("            C44 C45 C46")
    print("                C55 C56")
    print("                    C66")
    
    # For this example, let's provide a template
    print("\nUsing template for orthotropic material...")
    print("Modify the values in the script for your material\n")
    
    # Template orthotropic tensor
    C = np.array([
        [100.0,  30.0,  25.0,   0.0,   0.0,   0.0],
        [ 30.0, 120.0,  28.0,   0.0,   0.0,   0.0],
        [ 25.0,  28.0, 150.0,   0.0,   0.0,   0.0],
        [  0.0,   0.0,   0.0,  40.0,   0.0,   0.0],
        [  0.0,   0.0,   0.0,   0.0,  45.0,   0.0],
        [  0.0,   0.0,   0.0,   0.0,   0.0,  50.0]
    ])
    
    S = stiffness_to_compliance(C)
    constants = compliance_to_engineering_constants(S)
    valid, errors = validate_orthotropic_symmetry(constants)
    
    print_summary(C, constants, valid, errors)
    
    return C, constants


def read_from_lammps_output(filename):
    """
    Read elastic constants from LAMMPS output
    
    LAMMPS elastic constant calculation outputs a 6x6 matrix
    This function parses it and converts to engineering constants
    
    Parameters:
    -----------
    filename : str
        Path to LAMMPS output file containing elastic constants
    """
    # This is a template - adjust parsing based on your LAMMPS output format
    print(f"\nReading from LAMMPS output: {filename}")
    print("(Template function - adjust parsing for your output format)")
    
    # Example LAMMPS output format:
    # C11 C12 C13 C14 C15 C16
    # C12 C22 C23 C24 C25 C26
    # ...
    
    # You would implement parsing here
    # For now, return a template
    pass


def save_to_file(constants, filename='elastic_constants.txt'):
    """
    Save engineering constants to file
    """
    with open(filename, 'w') as f:
        f.write("# Engineering Constants for Orthotropic Material\n")
        f.write("# Generated from elastic tensor conversion\n\n")
        
        f.write(f"E1 = {constants['E1']:.6f} GPa\n")
        f.write(f"E2 = {constants['E2']:.6f} GPa\n")
        f.write(f"E3 = {constants['E3']:.6f} GPa\n\n")
        
        f.write(f"nu12 = {constants['nu12']:.6f}\n")
        f.write(f"nu13 = {constants['nu13']:.6f}\n")
        f.write(f"nu23 = {constants['nu23']:.6f}\n\n")
        
        f.write(f"G12 = {constants['G12']:.6f} GPa\n")
        f.write(f"G13 = {constants['G13']:.6f} GPa\n")
        f.write(f"G23 = {constants['G23']:.6f} GPa\n\n")
        
        f.write("# For Abaqus (in MPa):\n")
        f.write("# *Elastic, type=ORTHOTROPIC\n")
        f.write(f"# {format_for_abaqus(constants, unit='MPa')}\n")
    
    print(f"\nConstants saved to: {filename}")


# Main execution
if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("ELASTIC TENSOR TO ENGINEERING CONSTANTS CONVERTER")
    print("="*70)
    print("\nThis script converts 6x6 stiffness tensors (from MD/DFT)")
    print("to engineering constants needed for Abaqus/FE codes")
    
    # Example 1: Aragonite
    C_arag, const_arag = aragonite_example()
    save_to_file(const_arag, 'aragonite_constants.txt')
    
    # Example 2: BCC Iron
#    C_bcc, const_bcc = bcc_iron_example()
#    save_to_file(const_bcc, 'bcc_iron_constants.txt')
    
    # Example 3: Custom input (template)
    # C_custom, const_custom = custom_tensor_input()
    
    print("\n" + "="*70)
    print("USAGE NOTES")
    print("="*70)
    print("""
1. Replace the stiffness matrix C with your values from MD/DFT
2. Run this script to get engineering constants
3. Copy the Abaqus line into your input file
4. Check that symmetry validation passes

For MD simulations:
- Use LAMMPS 'compute elastic' or 'fix deform' commands
- Average over multiple configurations for better statistics
- Check temperature dependence if needed

For DFT calculations:
- Use stress-strain method or finite differences
- Check convergence with k-points and energy cutoff
- Acoustic sum rule should be satisfied: C11+C12+C13=C12+C22+C23=C13+C23+C33
""")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)