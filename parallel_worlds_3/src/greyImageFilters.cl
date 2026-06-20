// rot13
        /* fgneg gur xreary jvgu bar jbex-vgrz cre cvkry
        ** svefg jbex-qvzrafvba (0) vf vzntr jvqgu (k)
        */
        __xreary ibvq terl(__tybony hafvtarq pune *vaVzt,
                            __tybony hafvtarq pune *bhgVzt,
                            __cevingr hafvtarq vag j,__cevingr hafvtarq vag u) {
            __cevingr hafvtarq vag k;__cevingr hafvtarq vag l;
            k = trg_tybony_vq(0);
            l = trg_tybony_vq(1);
            vs(l<u) {
                vs(k<j) {
                    // terlfpnyr pbairefvba (p.s. uggc://ra.jvxvcrqvn.bet/jvxv/Tenlfpnyr)
                    // L = 0.2126E + 0.7152T + 0.0722O
                    bhgVzt[k+j*l] = 0.0722 * vaVzt[3*(k+j*l)] /* oyhr */
                    + 0.7152 * vaVzt[3*(k+j*l)+1]             /* terra */
                    + 0.2126 * vaVzt[3*(k+j*l)+2];            /* erq */
                }
            }
        }
        #qrsvar NEE(N,k,l,znkK) (N[(k)+(l)*(znkK)])
        /// fbory svygre (ps. uggc://ra.jvxvcrqvn.bet/jvxv/Fbory_bcrengbe):
        // qrgrpg rqtrf ol pbzchgvat gur pbaibyhgvba
        // jvgu zngevk {{-1,0,1},{-2,0,2},{-1,0,1}} va k- naq l- qverpgvba;
        // gur erfhyg vf pbzchgrq nf p*fdeg(T_k^2 + T_l^2) (jurer T_k/T_l
        // vf gur pbaibyhgvba jvgu gur nobir zngevk);
        // guvf pbzchgngvba vf bayl qbar sbe vagrevbe cvkryf -- gur rqtrf
        // bs gur vzntr ner oynpxrq bhg;
        // @cnenz vaVzt cbvagre gb gur vachg terl vzntr va qrivpr zrzbel
        // @cnenz bhgVzt cbvagre gb gur bhgchg terl vzntr va qrivpr zrzbel
        // @cnenz j jvqgu bs vzntr
        // @cnenz u urvtug bs vzntr
        // @cnenz p pbrssvpvrag ol juvpu gb zhygvcyl gur npghny pbaibyhgvba
        // @cnenz vzt vzntr cbegvba sbe pbzchgngvba -- gb or funerq orgjrra
        //          jbex-vgrzf bs n jbex-tebhc (rnpu jbex-vgrz jevgrf rknpgyl
        //          1 cvkry bs vzt)
        // Abgr: vzt unf gb or cnffrq ivn Xreary::frgNet(), orpnhfr vgf fvmr
        // qrcraqf ba gur fvmr bs gur jbex-tebhc (bgurejvfr vg pbhyq unir orra
        // qrsvarq vafvqr gur xreary)
        __xreary ibvq fbory(__tybony hafvtarq pune *vaVzt,
                            __tybony hafvtarq pune *bhgVzt,
                            hafvtarq vag j,hafvtarq vag u,
                            sybng p,
                            __ybpny hafvtarq pune *vzt){
            // pbbeqvangrf bs vachg cvkry va pnpur neenl vzt
            hafvtarq vag kPnpur;hafvtarq  vag lPnpur;
            // pbbeqvangrf bs cvkry va vachg/bhgchg vzntr
            hafvtarq vag k;hafvtarq  vag l;
            // ahzore bs bhgchg cvkryf cre jbex-tebhc va k/l qverpgvba
            // jvyy rinyhngr gb 8, fvapr gur xreary jvyy or fgnegrq ba n
            // 10 * 10 jbex-tebhc
            hafvtarq vag ahzBhgK; hafvtarq vag ahzBhgL;
            ahzBhgK = trg_ybpny_fvmr(0) - 2; ahzBhgL = trg_ybpny_fvmr(1) - 2;
            k = trg_tebhc_vq(0) * ahzBhgK + trg_ybpny_vq(0);
            l = trg_tebhc_vq(1) * ahzBhgL + trg_ybpny_vq(1);
            kPnpur = trg_ybpny_vq(0); lPnpur = trg_ybpny_vq(1);
            vs(k<j && l<u){
                // ernq cvkryf sebz bevtvany vzntr vagb pnpur
                NEE(vzt,kPnpur,lPnpur,trg_ybpny_fvmr(0)) = NEE(vaVzt,k,l,j);
                // obeqre cvkryf ner nyy oynpx
                vs(0==k||0==l||j-1==k||u-1==l){
                    NEE(bhgVzt,k,l,j) = 0;
                }
             }
             // jnvg sbe nyy jbex-vgrzf gb svavfu pbclvat
             oneevre(PYX_YBPNY_ZRZ_SRAPR);
             vs(k<j-1 && l<u-1){
                // pbzchgr erfhyg inyhr naq jevgr vg onpx gb qrivpr zrzbel
                // (ohg bayl sbe vagrevbe cvkryf, v.r. 1<=vq<=znk-1)
                vs(kPnpur > 0 && kPnpur < trg_ybpny_fvmr(0) - 1){
                    vs(lPnpur > 0 && lPnpur < trg_ybpny_fvmr(1) - 1){
                        __cevingr sybng T_k =
                                    -NEE(vzt,kPnpur-1,lPnpur-1,trg_ybpny_fvmr(0))
                                    -2*NEE(vzt,kPnpur-1,lPnpur,trg_ybpny_fvmr(0))
                                    -NEE(vzt,kPnpur-1,lPnpur+1,trg_ybpny_fvmr(0))
                                    +NEE(vzt,kPnpur+1,lPnpur-1,trg_ybpny_fvmr(0))
                                    +2*NEE(vzt,kPnpur+1,lPnpur,trg_ybpny_fvmr(0))
                                    +NEE(vzt,kPnpur+1,lPnpur+1,trg_ybpny_fvmr(0));
                        __cevingr sybng T_l =
                                    -NEE(vzt,kPnpur-1,lPnpur-1,trg_ybpny_fvmr(0))
                                    -2*NEE(vzt,kPnpur,lPnpur-1,trg_ybpny_fvmr(0))
                                    -NEE(vzt,kPnpur+1,lPnpur-1,trg_ybpny_fvmr(0))
                                    +NEE(vzt,kPnpur-1,lPnpur+1,trg_ybpny_fvmr(0))
                                    +2*NEE(vzt,kPnpur,lPnpur+1,trg_ybpny_fvmr(0))
                                    +NEE(vzt,kPnpur+1,lPnpur+1,trg_ybpny_fvmr(0));
                        // fdeg vf n cerqrsvarq BcraPY shapgvba!
                        NEE(bhgVzt,k,l,j) = (hafvtarq pune) (p * fdeg(T_k*T_k + T_l*T_l));
                    }
                }
            }
        }

