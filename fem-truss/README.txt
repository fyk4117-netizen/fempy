TRUSS.EXE v1.0 - USER GUIDE
FEM bar / truss element tutor (2D and 3D)
Bonus Assignment #1, Finite Element Analysis

WHAT IT IS
A step-by-step FEM tutor. It walks a truss problem from geometry to stresses
and shows every matrix along the way. All math is numpy. No AI at runtime.

FILES
  app.py                  the whole app (about 230 lines)
  requirements.txt        python packages
  .streamlit/config.toml  green-on-black terminal theme
  demo_report_ex3.9.txt   full solution printout for the demo example

ACCESS: ONLINE LINK (free, about 5 min)
  1. Make a public GitHub repo. Upload app.py, requirements.txt and the
     .streamlit folder (keep the folder name, including the dot).
  2. Go to share.streamlit.io, sign in with GitHub.
  3. Create app > pick the repo > branch main > main file app.py > Deploy.
  4. You get a link like https://<name>.streamlit.app. Submit that link.

ACCESS: RUN LOCALLY
  pip install -r requirements.txt
  streamlit run app.py
  Browser opens at http://localhost:8501

HOW TO USE
  Sidebar > LOAD PROBLEM   pick a preset:
      EX 3.9 (Logan) 3D space truss   (the demo / test case)
      2D three-bar truss
      2D seven-bar bridge
  Sidebar > GO TO STEP     jump to any of the 6 steps, or use PREV / NEXT.
  INPUT DECK (top)         edit any cell. Row n = node n.
      x, y, z in m. fix_x/y/z = checkbox, ticked means d = 0 there.
      Fx/Fy/Fz in kN. Elements: node_i, node_j, A in m^2, E in GPa.
      Add rows with the + at the bottom of a table. Set dimension 2 or 3.
      Everything recalculates instantly.

THE 6 STEPS (map to the grading rubric)
  1 PROBLEM          plot of nodes, members, supports (red), loads (cyan).
                     DOF table, connectivity, L and direction cosines.
  2 ELEMENT [k]      pick an element. L, C, local k-hat = AE/L[1 -1;-1 1],
                     transformation T*, lambda = C C^T, global k with DOF labels.
  3 ASSEMBLY [K]     DOF map per element. Slider adds elements one at a time;
                     green cells = entries the current element adds to.
  4 BOUNDARY COND.   fixed vs free DOFs, partitioned equations,
                     reduced [Kff] and {Ff}.
  5 DISPLACEMENTS    [K]{d}={F}. Expand the Gaussian elimination log to see
                     every row operation. Results in m and mm, checked against
                     numpy. Deformed shape plot with magnification slider.
  6 REACTIONS+STRESS R = K d - F at fixed DOFs, equilibrium check, strain,
                     stress in MPa, member force, TENSION / COMPRESSION.
                     SAVE REPORT.TXT downloads the full solution.

DEMO EXAMPLE: Logan Example 3.9 (3D space truss)
  Node 1 (12,-3,-4) m free, 20 kN in +x. Nodes 2 (0,0,0), 3 (12,-3,-7),
  4 (14,6,0) pinned. 3 bars, A = 10e-4 m^2, E = 210 GPa.
                 TRUSS.EXE        textbook
  d1x            1.3837e-3 m      1.383e-3 m
  d1y           -5.157e-5 m      -5.119e-5 m (book rounds lambda to 3 places)
  d1z            6.015e-5 m       6.015e-5 m
  sigma(1)       20.53 MPa  T     20.51 MPa
  sigma(2)        4.21 MPa  T      4.21 MPa
  sigma(3)       -5.29 MPa  C     -5.29 MPa

SIGN CONVENTION
  Element local axis runs node_i to node_j. Positive stress = tension.
