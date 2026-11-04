# Data freeze 2026-11-04

- Run: 2026-11-04 23:41 UTC on branch freeze-2026-11-04, code at 13c4e1c
- Data: monthly statistics run through the last complete month before the run; releases after it
  (e.g. October payrolls on 6 Nov, October CPI on 10 Nov) are in the narrative, not the models.
- Bloomberg caches (licensed, not committed):
  - hy_oas           last obs 2026-10-23
  - ism_new_orders   last obs 2026-09-30
  - ism_prices       last obs 2026-09-30
  - ism_inventories  last obs 2026-09-30
  - val_pe           last obs 2026-09-30
  - val_eps          last obs 2026-09-30
- Cap check:
  - Defensive active weight: Current +10.3pp, Base +9.4pp
  - Cap does not bind: Base is the recommendation as pre-registered
- Deviations from pre-registration: the post-hoc variant Base capped in src/weighting.py (team cap on the defensive overweight); it is the recommendation only if the cap check above says it binds.

## Download caches
```
                  file                      source   last obs       downloaded      per
            AWHMAN.csv                        FRED 2026-09-01 2026-11-04 23:38 manifest
               BAA.csv                        FRED 2026-10-01 2026-11-04 23:38 manifest
      BAMLH0A0HYM2.csv                        FRED 2026-11-03 2026-11-04 23:38 manifest
          CPIAUCSL.csv                        FRED 2026-09-01 2026-11-04 23:38 manifest
          CPILFESL.csv                        FRED 2026-09-01 2026-11-04 23:38 manifest
        DCOILWTICO.csv                        FRED 2026-11-03 2026-11-04 23:38 manifest
               DFF.csv                        FRED 2026-11-03 2026-11-04 23:38 manifest
            DFII10.csv                        FRED 2026-11-03 2026-11-04 23:38 manifest
             DGS10.csv                        FRED 2026-11-03 2026-11-04 23:38 manifest
              DGS2.csv                        FRED 2026-11-03 2026-11-04 23:38 manifest
          DTWEXBGS.csv                        FRED 2026-10-30 2026-11-04 23:38 manifest
          FEDFUNDS.csv                        FRED 2026-10-01 2026-11-04 23:38 manifest
GACDFSA066MSFRBPHI.csv                        FRED 2026-10-01 2026-11-04 23:38 manifest
 GACDISA066MSFRBNY.csv                        FRED 2026-10-01 2026-11-04 23:38 manifest
              GS10.csv                        FRED 2026-10-01 2026-11-04 23:38 manifest
              ICSA.csv                        FRED 2026-10-24 2026-11-04 23:38 manifest
            INDPRO.csv                        FRED 2026-09-01 2026-11-04 23:38 manifest
              MICH.csv                        FRED 2026-09-01 2026-11-04 23:38 manifest
              NFCI.csv                        FRED 2026-10-30 2026-11-04 23:38 manifest
NOCDFSA066MSFRBPHI.csv                        FRED 2026-10-01 2026-11-04 23:38 manifest
            PAYEMS.csv                        FRED 2026-09-01 2026-11-04 23:38 manifest
          PCEPILFE.csv                        FRED 2026-09-01 2026-11-04 23:38 manifest
            PERMIT.csv                        FRED 2026-09-01 2026-11-04 23:38 manifest
PPCDFSA066MSFRBPHI.csv                        FRED 2026-10-01 2026-11-04 23:38 manifest
         RRPONTSYD.csv                        FRED 2026-11-04 2026-11-04 23:38 manifest
             RRSFS.csv                        FRED 2026-09-01 2026-11-04 23:38 manifest
            T10YIE.csv                        FRED 2026-11-04 2026-11-04 23:38 manifest
             T5YIE.csv                        FRED 2026-11-04 2026-11-04 23:38 manifest
            T5YIFR.csv                        FRED 2026-11-04 2026-11-04 23:38 manifest
             TB3MS.csv                        FRED 2026-10-01 2026-11-04 23:38 manifest
           UMCSENT.csv                        FRED 2026-09-01 2026-11-04 23:38 manifest
             USREC.csv                        FRED 2026-10-01 2026-11-04 23:38 manifest
            VIXCLS.csv                        FRED 2026-11-03 2026-11-04 23:38 manifest
            VXVCLS.csv                        FRED 2026-11-03 2026-11-04 23:38 manifest
             WALCL.csv                        FRED 2026-10-28 2026-11-04 23:38 manifest
           WTREGEN.csv                        FRED 2026-10-28 2026-11-04 23:38 manifest
    prices_monthly.csv         Yahoo (sector ETFs) 2026-11-30 2026-11-04 23:38 manifest
     market_prices.csv Yahoo (SPY/IWM/gold/copper) 2026-11-30 2026-11-04 23:38 manifest
   holdings_prices.csv            Yahoo (VGT/SCHH) 2026-11-30 2026-11-04 23:38 manifest
        ff_factors.csv                  Ken French 2026-09-30 2026-11-04 23:38 manifest
```
