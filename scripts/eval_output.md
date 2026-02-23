```
✅  Loaded 30 valid error codes from JSON.

Loading EasyOCR model from D:\Samsung Prism Internship\25ST32MS_Troubleshoot_Helper_Revolutionizing_Device_Diagnostics\config\assets\models...
  ❌  3E 1.jpg            GT=3E      Raw=0         Final=0       (NONE)
  ❌  3E.jpg              GT=3E      Raw=          Final=        (NONE)
  ❌  4C.png              GT=4C      Raw=p         Final=p       (NONE)
  ✅  dC 1.jpg            GT=dC      Raw=dC        Final=dc      (DIRECT)
  ❌  dC 2.jpg            GT=dC      Raw=IpE       Final=IpE     (NONE)
  ❌  dC 3.jpg            GT=dC      Raw=C         Final=C       (NONE)
  ❌  dC 4.jpg            GT=dC      Raw=1         Final=1       (NONE)
  ❌  dC 5.jpg            GT=dC      Raw=Ol Fte    Final=OlFte   (NONE)
  ❌  dC 6.jpg            GT=dC      Raw=Iin       Final=Iin     (NONE)
  ❌  dC 7.jpg            GT=dC      Raw=          Final=        (NONE)
  ❌  dC 8.jpg            GT=dC      Raw=          Final=        (NONE)
  ❌  dC.jpg              GT=dC      Raw=Q         Final=Q       (NONE)
  ❌  dE 1.jpg            GT=dE      Raw=dBerrorcode 88  Final=dBerrorcode88  (NONE)
  ❌  dE 2.jpg            GT=dE      Raw=OW0EcC 5  Final=OW0EcC5  (NONE)
  ❌  dE.jpg              GT=dE      Raw=dEerrorcode liuts 5 Qeu  Final=dEerrorcodeliuts5Qeu  (NONE)
  ❌  dS 1.jpg            GT=dS      Raw=85        Final=86      (FUZZY)
  ❌  dS.jpg              GT=dS      Raw=35        Final=E3      (FUZZY)
  ❌  E2 1.jpg            GT=E2      Raw=Ez        Final=LE      (FUZZY)
  ❌  E2 2.jpg            GT=E2      Raw=          Final=        (NONE)
  ✅  E2 3.jpg            GT=E2      Raw=E2        Final=E2      (DIRECT)
  ❌  E2 4.jpg            GT=E2      Raw=HOWtOFIXE2vERRORIN MYWASHER  Final=HOWtOFIXE2vERRORINMYWASHER  (NONE)
  ❌  E2.jpg              GT=E2      Raw=62 Vat doorswitch  Final=62Vatdoorswitch  (NONE)
  ✅  E3 1.jpg            GT=E3      Raw=E3        Final=E3      (DIRECT)
  ❌  E3 2.jpg            GT=E3      Raw=83        Final=86      (FUZZY)
  ❌  E3 3.jpg            GT=E3      Raw=          Final=        (NONE)
  ❌  E3.jpg              GT=E3      Raw=5         Final=5       (NONE)
  ❌  E3A.png             GT=E3A     Raw=Iwa       Final=Iwa     (NONE)
  ❌  E7.jpg              GT=E7      Raw=E         Final=E       (NONE)
  ❌  F24.png             GT=F24     Raw=et 1I     Final=et1I    (NONE)
  ❌  FL 2.jpg            GT=FL      Raw=          Final=        (NONE)
  ❌  FL.jpg              GT=FL      Raw=ehlnt     Final=ehlnt   (NONE)
  ❌  H20.png             GT=H20     Raw=Hod Ktean Dat  Final=HodKteanDat  (NONE)
  ❌  LE 1.jpg            GT=LE      Raw=          Final=        (NONE)
  ❌  LE 2.jpg            GT=LE      Raw=LB        Final=LO      (FUZZY)
  ❌  LE.jpg              GT=LE      Raw=0 QE ERROR  Final=0QEERROR  (NONE)
  ❌  LO.jpg              GT=LO      Raw=4         Final=4       (NONE)
  ❌  nD.jpg              GT=nD      Raw=68        Final=86      (FUZZY)
  ✅  nF 1.jpg            GT=nF      Raw=1F        Final=nF      (FUZZY)
  ✅  nF 2.jpg            GT=nF      Raw=IF        Final=nF      (FUZZY)
  ❌  nF 3.jpg            GT=nF      Raw=SatSUNJI  Final=SatSUNJI  (NONE)
  ❌  nF.jpg              GT=nF      Raw=Nff       Final=nF1     (FUZZY)
  ❌  PF 1.jpg            GT=PF      Raw=Erta PF   Final=ErtaPF  (NONE)
  ❌  PF 2.jpg            GT=PF      Raw=Te        Final=LE      (FUZZY)
  ❌  PF.jpg              GT=PF      Raw=          Final=        (NONE)
  ✅  SUD (2).jpg         GT=SUD     Raw=suo       Final=SUd     (SUBSTITUTION)
  ✅  SUd 2.jpg           GT=SUd     Raw=TUd       Final=SUd     (FUZZY)
  ✅  SUd.jpg             GT=SUd     Raw=TUd       Final=SUd     (FUZZY)

════════════════════════════════════════════════════════════════════════
            BATCH  EVALUATION  REPORT
════════════════════════════════════════════════════════════════════════
  Overall Accuracy   : 17.02%
  Correct / Total    : 8 / 47
  Incorrect          : 39
  Skipped (no label) : 0
  Total time         : 39.91s
────────────────────────────────────────────────────────────────────────
  FAILURE LOG  (39 errors)
────────────────────────────────────────────────────────────────────────
    # │ Filename           │ Ground Truth │ Raw OCR      │ Pipeline     │ Final Pred   │ Step
  ────┼────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼─────────────
    1 │ 3E 1.jpg           │ 3E           │ 0            │ 0            │ 0            │ NONE
    2 │ 3E.jpg             │ 3E           │              │              │              │ NONE
    3 │ 4C.png             │ 4C           │ p            │ p            │ p            │ NONE
    4 │ dC 2.jpg           │ dC           │ IpE          │ IpE          │ IpE          │ NONE
    5 │ dC 3.jpg           │ dC           │ C            │ C            │ C            │ NONE
    6 │ dC 4.jpg           │ dC           │ 1            │ 1            │ 1            │ NONE
    7 │ dC 5.jpg           │ dC           │ Ol Fte       │ OF           │ OlFte        │ NONE
    8 │ dC 6.jpg           │ dC           │ Iin          │ Iin          │ Iin          │ NONE
    9 │ dC 7.jpg           │ dC           │              │              │              │ NONE
   10 │ dC 8.jpg           │ dC           │              │              │              │ NONE
   11 │ dC.jpg             │ dC           │ Q            │ Q            │ Q            │ NONE
   12 │ dE 1.jpg           │ dE           │ dBerrorcode 88 │ bE           │ dBerrorcode88 │ NONE
   13 │ dE 2.jpg           │ dE           │ OW0EcC 5     │ OW0          │ OW0EcC5      │ NONE
   14 │ dE.jpg             │ dE           │ dEerrorcode liuts 5 Qeu │ dE           │ dEerrorcodeliuts5Qeu │ NONE
   15 │ dS 1.jpg           │ dS           │ 85           │ 85           │ 86           │ FUZZY
   16 │ dS.jpg             │ dS           │ 35           │ 35           │ E3           │ FUZZY
   17 │ E2 1.jpg           │ E2           │ Ez           │ Ez           │ LE           │ FUZZY
   18 │ E2 2.jpg           │ E2           │              │              │              │ NONE
   19 │ E2 4.jpg           │ E2           │ HOWtOFIXE2vERRORIN MYWASHER │ HOW          │ HOWtOFIXE2vERRORINMYWASHER │ NONE
   20 │ E2.jpg             │ E2           │ 62 Vat doorswitch │ 62V          │ 62Vatdoorswitch │ NONE
   21 │ E3 2.jpg           │ E3           │ 83           │ 83           │ 86           │ FUZZY
   22 │ E3 3.jpg           │ E3           │              │              │              │ NONE
   23 │ E3.jpg             │ E3           │ 5            │ 5            │ 5            │ NONE
   24 │ E3A.png            │ E3A          │ Iwa          │ Iwa          │ Iwa          │ NONE
   25 │ E7.jpg             │ E7           │ E            │ E            │ E            │ NONE
   26 │ F24.png            │ F24          │ et 1I        │ et1          │ et1I         │ NONE
   27 │ FL 2.jpg           │ FL           │              │              │              │ NONE
   28 │ FL.jpg             │ FL           │ ehlnt        │ ehl          │ ehlnt        │ NONE
   29 │ H20.png            │ H20          │ Hod Ktean Dat │ Hod          │ HodKteanDat  │ NONE
   30 │ LE 1.jpg           │ LE           │              │              │              │ NONE
   31 │ LE 2.jpg           │ LE           │ LB           │ LB           │ LO           │ FUZZY
   32 │ LE.jpg             │ LE           │ 0 QE ERROR   │ 0QE          │ 0QEERROR     │ NONE
   33 │ LO.jpg             │ LO           │ 4            │ 4            │ 4            │ NONE
   34 │ nD.jpg             │ nD           │ 68           │ 68           │ 86           │ FUZZY
   35 │ nF 3.jpg           │ nF           │ SatSUNJI     │ Sat          │ SatSUNJI     │ NONE
   36 │ nF.jpg             │ nF           │ Nff          │ nF           │ nF1          │ FUZZY
   37 │ PF 1.jpg           │ PF           │ Erta PF      │ Ert          │ ErtaPF       │ NONE
   38 │ PF 2.jpg           │ PF           │ Te           │ Te           │ LE           │ FUZZY
   39 │ PF.jpg             │ PF           │              │              │              │ NONE
────────────────────────────────────────────────────────────────────────
════════════════════════════════════════════════════════════════════════

```
