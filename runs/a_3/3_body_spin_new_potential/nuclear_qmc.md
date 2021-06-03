# Nuclear QMC Run
## Log File
/home/nbrawand/computation/argonne/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0001,
        "n_blocks": 5,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 5,
        "n_optimization_steps": 1000000,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+1)
## Wave Function Parameters
creating wave function parameters file: /home/nbrawand/computation/argonne/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | 11.339116879189344 | 0.3704419254494851
optimization step | 1 | 10.883375890163897 | 0.21388585486391334
optimization step | 2 | 10.738171358965978 | 0.38497625655115625
optimization step | 3 | 10.27023139624004 | 0.4167562315259483
optimization step | 4 | 10.149383496153444 | 0.15964862081169545
optimization step | 5 | 10.14972589033538 | 0.1262214866121296
optimization step | 6 | 9.683221169593585 | 0.23959469818070406
optimization step | 7 | 9.48984420764028 | 0.21858458500600383
optimization step | 8 | 9.658787580715202 | 0.07628084941286635
optimization step | 9 | 9.558995047035067 | 0.08030971980743229
optimization step | 10 | 9.382301008417697 | 0.08534137137931802
optimization step | 11 | 8.992353256607746 | 0.2071572910774629
optimization step | 12 | 9.163787749045266 | 0.16318962360429357
optimization step | 13 | 8.975638528048817 | 0.11585853923635409
optimization step | 14 | 9.039212437215907 | 0.09463206996506254
optimization step | 15 | 8.91293830402833 | 0.11530557675144261
optimization step | 16 | 9.035728624550055 | 0.02786159181386712
optimization step | 17 | 8.881254276325382 | 0.04893587757509473
optimization step | 18 | 9.033404185648287 | 0.05479324102550314
optimization step | 19 | 8.827849438368121 | 0.05046152882498339
optimization step | 20 | 8.876812480986448 | 0.05263578733050245
optimization step | 21 | 8.632946075357925 | 0.028099383015734745
optimization step | 22 | 8.866090712690717 | 0.028689037052456286
optimization step | 23 | 8.780703702593254 | 0.045153805463548996
optimization step | 24 | 8.743224094724884 | 0.03918485014801235
optimization step | 25 | 8.79088495429922 | 0.08371377722328832
optimization step | 26 | 8.874016544690036 | 0.0255029029763231
optimization step | 27 | 8.814993402521486 | 0.07581571821318311
optimization step | 28 | 8.70632045627119 | 0.06432641888297824
optimization step | 29 | 8.627125509555643 | 0.03574796178480601
optimization step | 30 | 8.634621429298374 | 0.08241964965142537
optimization step | 31 | 8.670412712795025 | 0.03749208371315279
optimization step | 32 | 8.752557966410915 | 0.05153009140260822
optimization step | 33 | 8.765507864333198 | 0.04393813379627019
optimization step | 34 | 8.62081600650249 | 0.04066427133294597
optimization step | 35 | 8.662106208650567 | 0.03898347137499133
optimization step | 36 | 8.542206080319886 | 0.06187641087843271
optimization step | 37 | 8.55664831008024 | 0.03571227235018851
optimization step | 38 | 8.501382799829813 | 0.08044157304257478
optimization step | 39 | 8.565565405285728 | 0.06153545531313715
optimization step | 40 | 8.640805141258973 | 0.06548302654288632
optimization step | 41 | 8.429065934543171 | 0.06213119506888338
optimization step | 42 | 8.432328763648684 | 0.04714278105255115
optimization step | 43 | 8.386067085275487 | 0.03489636713547013
optimization step | 44 | 8.53014127966428 | 0.04096884028474506
optimization step | 45 | 8.501873175239526 | 0.04772944280488663
optimization step | 46 | 8.409906015589577 | 0.03731476280933481
optimization step | 47 | 8.491686419523468 | 0.05015008732435202
optimization step | 48 | 8.373845458026754 | 0.048501594674651254
optimization step | 49 | 8.360392401714497 | 0.0398774831132484
optimization step | 50 | 8.41935103703438 | 0.0435413435059417
optimization step | 51 | 8.332412716281286 | 0.033462105312171884
optimization step | 52 | 8.340739555678885 | 0.046606001575052215
optimization step | 53 | 8.316335870592743 | 0.036039617747834096
optimization step | 54 | 8.339499121440376 | 0.056023621830689055
optimization step | 55 | 8.304838301927807 | 0.04440156554595008
optimization step | 56 | 8.216783308086672 | 0.05501323948418656
optimization step | 57 | 8.336062616964252 | 0.046467126250006964
optimization step | 58 | 8.363965563017254 | 0.041145791538277035
optimization step | 59 | 8.294132910684699 | 0.031047551806040255
optimization step | 60 | 8.237306355461962 | 0.017430562866298512
optimization step | 61 | 8.3447431837461 | 0.05793956673124459
optimization step | 62 | 8.238438734291188 | 0.031034854023039876
optimization step | 63 | 8.2578252071901 | 0.012872475680819841
optimization step | 64 | 8.22857799849047 | 0.03470156079969769
optimization step | 65 | 8.225174392124796 | 0.04256852183730177
optimization step | 66 | 8.211772706787679 | 0.05808267009610704
optimization step | 67 | 8.221272807317103 | 0.0184554989510119
optimization step | 68 | 8.131601085861565 | 0.05629439030816719
optimization step | 69 | 8.162185879951622 | 0.04220499528612944
optimization step | 70 | 8.179595730692911 | 0.02622544544304837
optimization step | 71 | 8.097950764332918 | 0.05076826525642979
optimization step | 72 | 8.136281826963563 | 0.06255837071459282
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0001,
        "n_blocks": 5,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 5,
        "n_optimization_steps": 1000000,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+1)
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -2.006179806109806 | 0.1475126093330518
optimization step | 1 | -1.9898765934846554 | 0.05870418939163072
optimization step | 2 | -2.3994620843405206 | 0.08407052207218209
optimization step | 3 | -2.0710888025448577 | 0.05827326330361947
optimization step | 4 | -2.1254854168336577 | 0.12586536392326209
optimization step | 5 | -2.432239015896919 | 0.09914843886345955
optimization step | 6 | -2.341749328867883 | 0.045636406755717195
optimization step | 7 | -2.604647892839953 | 0.12866300282690551
optimization step | 8 | -2.54704843849171 | 0.14224330490849774
optimization step | 9 | -2.5755048793087774 | 0.11035679973811317
optimization step | 10 | -2.693672115768292 | 0.07230124201564486
optimization step | 11 | -2.6334157173420825 | 0.13095576999925
optimization step | 12 | -2.6519583250918055 | 0.04399010052582948
optimization step | 13 | -2.8975614031243886 | 0.10041280437250792
optimization step | 14 | -2.7342427773750737 | 0.06357206099739274
optimization step | 15 | -2.8384033241614 | 0.053116653986338504
optimization step | 16 | -2.8739460003225803 | 0.11252846636714088
optimization step | 17 | -3.0573179254147878 | 0.06302124575094148
optimization step | 18 | -3.002061825075177 | 0.06129936792261743
optimization step | 19 | -3.204402444106548 | 0.09958086539895475
optimization step | 20 | -2.809540946999542 | 0.04368082142707714
optimization step | 21 | -3.095848465738304 | 0.0848232554724191
optimization step | 22 | -3.1894319408505756 | 0.07509556934392839
optimization step | 23 | -3.188316126649256 | 0.11109450325500818
optimization step | 24 | -3.0240904980623364 | 0.11419425634314796
optimization step | 25 | -3.392090569577803 | 0.08163892817718249
optimization step | 26 | -3.3027025139657695 | 0.09498719671589716
optimization step | 27 | -3.29269969450333 | 0.03462093425119609
optimization step | 28 | -3.400039953217184 | 0.10802933581078283
optimization step | 29 | -3.539362027690718 | 0.11474372776397856
optimization step | 30 | -3.603952048054313 | 0.11196586518461277
optimization step | 31 | -3.452226044556631 | 0.06627458972377527
optimization step | 32 | -3.7104662583395416 | 0.034230012673512396
optimization step | 33 | -3.625758274894906 | 0.09201742468449897
optimization step | 34 | -3.551343236739746 | 0.0673276759124691
optimization step | 35 | -3.658708277638661 | 0.08781223707318557
optimization step | 36 | -3.7245125812025237 | 0.08317443182369565
optimization step | 37 | -3.7887945818060835 | 0.06240650165122199
optimization step | 38 | -3.6994851610994397 | 0.03568157292752286
optimization step | 39 | -3.7497557937902166 | 0.09348827839146304
optimization step | 40 | -3.877545809188934 | 0.05363183265851144
optimization step | 41 | -4.0803900446465855 | 0.09106456451283344
optimization step | 42 | -4.003143100386241 | 0.11863541518821401
optimization step | 43 | -3.8913236737714123 | 0.14605100850466168
optimization step | 44 | -3.9145663204256627 | 0.06450189244124498
optimization step | 45 | -4.191432603661713 | 0.08990128825027711
optimization step | 46 | -4.150490134519359 | 0.04720040878972375
optimization step | 47 | -4.316896195425927 | 0.12163434778846904
optimization step | 48 | -4.237654856478015 | 0.14596517304930348
optimization step | 49 | -4.3105165698636885 | 0.12081862195202189
optimization step | 50 | -4.238232865066631 | 0.13538529213326908
optimization step | 51 | -4.458520580468108 | 0.07350375756836829
optimization step | 52 | -4.277833661676397 | 0.1007128391004148
optimization step | 53 | -4.250463969393553 | 0.14119728312854948
optimization step | 54 | -4.439819431521582 | 0.07874682946460242
optimization step | 55 | -4.370906365918549 | 0.0595485263402801
optimization step | 56 | -4.28607810381088 | 0.05811619800719876
optimization step | 57 | -4.46899804089063 | 0.07555568858063122
optimization step | 58 | -4.501256197175686 | 0.06006011178399805
optimization step | 59 | -4.590952924358191 | 0.06982259394489387
optimization step | 60 | -4.751876326986793 | 0.0567471919782881
optimization step | 61 | -4.612609216975647 | 0.08586792615725564
optimization step | 62 | -4.698137417334051 | 0.05097476817503552
optimization step | 63 | -4.835126868651059 | 0.06060901006498868
optimization step | 64 | -4.82191574677415 | 0.08752659047213641
optimization step | 65 | -4.789075928983879 | 0.09003615749489524
optimization step | 66 | -4.86157866159086 | 0.04064031930736577
optimization step | 67 | -4.8769407049131726 | 0.09852807330085406
optimization step | 68 | -4.7631630566405345 | 0.1355011693038293
optimization step | 69 | -4.872792460397645 | 0.11389513080223258
optimization step | 70 | -4.875985828280411 | 0.06509553887235213
optimization step | 71 | -4.745717180046578 | 0.09696065541003548
optimization step | 72 | -4.858398995715963 | 0.0681589447951862
optimization step | 73 | -4.928714710839616 | 0.10562236925236937
optimization step | 74 | -5.193923730558376 | 0.05351092062718651
optimization step | 75 | -4.897692843967444 | 0.044808677124087114
optimization step | 76 | -5.062274826353101 | 0.07102026754084488
optimization step | 77 | -5.0007281208587315 | 0.07421598288650727
optimization step | 78 | -5.107791138124065 | 0.0415118242430748
optimization step | 79 | -5.159601724728693 | 0.06838555139548705
optimization step | 80 | -5.034522934470745 | 0.060323244572944815
optimization step | 81 | -5.204648765125924 | 0.08318668414397587
optimization step | 82 | -5.191744412516565 | 0.08674772641654861
optimization step | 83 | -5.115601729249798 | 0.07030863802988781
optimization step | 84 | -5.264928968656089 | 0.052412451300368694
optimization step | 85 | -5.282858419938148 | 0.06121806047200377
optimization step | 86 | -5.339079656962275 | 0.059844368143339495
optimization step | 87 | -5.239429373776112 | 0.06809087505546688
optimization step | 88 | -5.286354847856484 | 0.04529719580234895
optimization step | 89 | -5.324231347467371 | 0.06849087464737563
optimization step | 90 | -5.426598926167527 | 0.11635146968118561
optimization step | 91 | -5.351899400744185 | 0.039532952112323966
optimization step | 92 | -5.480726905898946 | 0.029891886895321452
optimization step | 93 | -5.388144486976328 | 0.058902860136028516
optimization step | 94 | -5.454658931724572 | 0.04179345272101851
optimization step | 95 | -5.389141113917904 | 0.06484353883223057
optimization step | 96 | -5.416745121153292 | 0.08021326017473146
optimization step | 97 | -5.532738484731214 | 0.0606898783574276
optimization step | 98 | -5.628842091602188 | 0.042679617203415864
optimization step | 99 | -5.564503645712276 | 0.08735029601418676
optimization step | 100 | -5.549626080230366 | 0.06658893894994654
optimization step | 101 | -5.52539162044765 | 0.04747547732603067
optimization step | 102 | -5.585464835345962 | 0.0378067980856535
optimization step | 103 | -5.594869775258095 | 0.07521366326164458
optimization step | 104 | -5.56597683782705 | 0.05812136961163517
optimization step | 105 | -5.632775178441194 | 0.061382348351416705
optimization step | 106 | -5.714734049944049 | 0.060658302400873707
optimization step | 107 | -5.671357848053887 | 0.04591972270467448
optimization step | 108 | -5.776644035710126 | 0.02655604069077614
optimization step | 109 | -5.645599720352979 | 0.05575764588586941
optimization step | 110 | -5.780078798057559 | 0.05323091603120989
optimization step | 111 | -5.769704945113854 | 0.03178293843011499
optimization step | 112 | -5.70418611606202 | 0.039328549853695056
optimization step | 113 | -5.84815569272613 | 0.04320073995288792
optimization step | 114 | -5.882818791254508 | 0.06420198225406576
optimization step | 115 | -5.930617552125062 | 0.034731366073145496
optimization step | 116 | -5.795953484953435 | 0.07906104514884031
optimization step | 117 | -5.845758746587125 | 0.0625505585861928
optimization step | 118 | -6.009055506143023 | 0.06487433981073168
optimization step | 119 | -6.013886273517139 | 0.04817809011162034
optimization step | 120 | -5.884593758695064 | 0.05601995050681491
optimization step | 121 | -5.924540819534805 | 0.07521861353893583
optimization step | 122 | -6.002325216429298 | 0.06330283051433046
optimization step | 123 | -6.061181958570341 | 0.09771824832235902
optimization step | 124 | -5.9807829936079475 | 0.08219122745478236
optimization step | 125 | -5.9760115483492 | 0.04701332086737104
optimization step | 126 | -6.001012349219551 | 0.07277070544914446
optimization step | 127 | -5.994979831959129 | 0.02713222363951643
optimization step | 128 | -5.986159075859581 | 0.07871395887970545
optimization step | 129 | -6.041992440098978 | 0.04449792067279563
optimization step | 130 | -5.952704387391019 | 0.05831699663266946
optimization step | 131 | -6.0115125242167515 | 0.06059158211543854
optimization step | 132 | -6.076365105139566 | 0.07062949355249523
optimization step | 133 | -6.1783759242102745 | 0.02967767094969014
optimization step | 134 | -6.105512601142144 | 0.06249105038213938
optimization step | 135 | -6.257746239941847 | 0.062409486001002074
optimization step | 136 | -6.201587458499203 | 0.06237822227401365
optimization step | 137 | -6.25821979416392 | 0.04557363306470926
optimization step | 138 | -6.255516068642182 | 0.02994292267168903
optimization step | 139 | -6.133843605035898 | 0.034568065906217214
optimization step | 140 | -6.246908189191858 | 0.0490800231590001
optimization step | 141 | -6.249796046652451 | 0.05045684463174134
optimization step | 142 | -6.195609609280929 | 0.03473671426503339
optimization step | 143 | -6.448642863046466 | 0.05347827191977778
optimization step | 144 | -6.21553291015088 | 0.03777235679221261
optimization step | 145 | -6.308058330985923 | 0.03391805207733146
optimization step | 146 | -6.48192549616798 | 0.0835756427591093
optimization step | 147 | -6.404505493469054 | 0.06774215470481008
optimization step | 148 | -6.364954238905021 | 0.04693711396691341
optimization step | 149 | -6.4306265392460205 | 0.07544373853532
optimization step | 150 | -6.360543550726289 | 0.05369054100542131
optimization step | 151 | -6.385970071857687 | 0.10160188999194619
optimization step | 152 | -6.467780993453782 | 0.024143480004690104
optimization step | 153 | -6.540615238407009 | 0.011849182080934724
optimization step | 154 | -6.484734660378261 | 0.03651723332100104
optimization step | 155 | -6.458997294589882 | 0.023541366279992065
optimization step | 156 | -6.541260974182076 | 0.04349266500133288
optimization step | 157 | -6.435737300086524 | 0.049141504652874195
optimization step | 158 | -6.555946812879297 | 0.05252272249237891
optimization step | 159 | -6.4656288599937515 | 0.04149452695176839
optimization step | 160 | -6.451438889392617 | 0.02282145987729853
optimization step | 161 | -6.528380355956062 | 0.07860735349635246
optimization step | 162 | -6.59962912279578 | 0.07284930218455403
optimization step | 163 | -6.652042265962083 | 0.03755467426096806
optimization step | 164 | -6.517938795755268 | 0.024171592073962148
optimization step | 165 | -6.5516414600393365 | 0.055454022360350366
optimization step | 166 | -6.582691644197487 | 0.03869769705606118
optimization step | 167 | -6.6453391728865 | 0.02472426887972291
optimization step | 168 | -6.6877789222683335 | 0.02959643786612402
optimization step | 169 | -6.787081318841841 | 0.051171666716355896
optimization step | 170 | -6.735163736463564 | 0.048286966651303664
optimization step | 171 | -6.683954167816346 | 0.0318281963292205
optimization step | 172 | -6.736767352482528 | 0.03394308868376221
optimization step | 173 | -6.7178375290701 | 0.05852539395091417
optimization step | 174 | -6.676254503537362 | 0.04597521943550652
optimization step | 175 | -6.677008639805189 | 0.052333224696419114
optimization step | 176 | -6.617750580736448 | 0.03404391335949585
optimization step | 177 | -6.710712072849731 | 0.01925000763203343
optimization step | 178 | -6.776024311665507 | 0.07097943848022767
optimization step | 179 | -6.767797499137832 | 0.053205713990327495
optimization step | 180 | -6.8592651867310455 | 0.05456900004425051
optimization step | 181 | -6.756531323531092 | 0.013584004297775487
optimization step | 182 | -6.859540468868426 | 0.054371056157879835
optimization step | 183 | -6.784182213937845 | 0.046973255824187904
optimization step | 184 | -6.825441459831586 | 0.028912935206905032
optimization step | 185 | -6.833101879642102 | 0.04532887269640696
optimization step | 186 | -6.8560324639295445 | 0.0740425888511797
optimization step | 187 | -6.82736604723633 | 0.04555841531735263
optimization step | 188 | -6.889425917995728 | 0.051524387410456736
optimization step | 189 | -6.9665580858687495 | 0.005792584452879515
optimization step | 190 | -6.856711167076982 | 0.016783733247641944
optimization step | 191 | -6.877441428423117 | 0.024469982594116563
optimization step | 192 | -6.928769346993404 | 0.04225579972599171
optimization step | 193 | -7.007745992905134 | 0.039656480565171764
optimization step | 194 | -6.870614966182883 | 0.018873555479812756
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.001,
        "n_blocks": 5,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 5,
        "n_optimization_steps": 1000000,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+1)
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -6.980613415402272 | 0.04800487727889697
optimization step | 1 | -7.070460093628247 | 0.031047276725559632
optimization step | 2 | -7.297022717799123 | 0.024194405417925113
optimization step | 3 | -7.369658830366047 | 0.03376044094586633
optimization step | 4 | -7.758776575235366 | 0.09272630426510223
optimization step | 5 | 11.882336125425377 | 0.1784775279128103
optimization step | 6 | 3.150260233944556 | 0.21996424851309074
optimization step | 7 | 2.493181297889884 | 0.16389202916364828
optimization step | 8 | 2.2453953294767794 | 0.08906209390358166
optimization step | 9 | 1.5375451050893887 | 0.061352925468713286
optimization step | 10 | 0.8473777423788672 | 0.10107737236366955
optimization step | 11 | 0.5993133320045284 | 0.070896010522642
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0005,
        "n_blocks": 5,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 5,
        "n_optimization_steps": 1000000,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+1)
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -6.935187590492762 | 0.033328903589959706
optimization step | 1 | -6.990347375348729 | 0.024595307623858788
optimization step | 2 | -7.095139933453451 | 0.021852512454446277
optimization step | 3 | -7.097936733708425 | 0.027420450522739576
optimization step | 4 | -7.2489002797024895 | 0.023304692776970535
optimization step | 5 | -7.4085502695977725 | 0.02385339668731471
optimization step | 6 | -7.4792100484451565 | 0.011426976733286463
optimization step | 7 | -7.377494372938256 | 0.03488039044118238
optimization step | 8 | -7.549762600870475 | 0.012311719181544199
optimization step | 9 | -7.569322907987363 | 0.03604926251059593
optimization step | 10 | -7.55808690780272 | 0.013581216665839875
optimization step | 11 | -7.634484338928902 | 0.03226536723372437
optimization step | 12 | -7.703616001149925 | 0.022439988902925343
optimization step | 13 | -7.7583770102069405 | 0.02911936529012222
optimization step | 14 | -7.698134509607124 | 0.018534769968520166
optimization step | 15 | -7.724461344906965 | 0.058066033970837
optimization step | 16 | -7.704764471877357 | 0.023756175362498758
optimization step | 17 | -7.724769644098416 | 0.03034426884131142
optimization step | 18 | -7.806943028648947 | 0.027070555601179054
optimization step | 19 | -7.795270524011062 | 0.03233631755566486
optimization step | 20 | -7.773260864224829 | 0.028584194354864024
optimization step | 21 | -7.851176477256409 | 0.015267389738277107
optimization step | 22 | -8.054722830187282 | 0.013153761370984353
optimization step | 23 | 5.211431128925764 | 0.22259747611216923
optimization step | 24 | -6.0644892983735215 | 0.23534397055906595
optimization step | 25 | -5.164328418084688 | 0.01750919200509485
optimization step | 26 | -5.482655936948528 | 0.08177910379346658
optimization step | 27 | -5.324729159829291 | 0.08280454112825406
optimization step | 28 | -5.578356384911783 | 0.03263676216826905
optimization step | 29 | -5.856178634619536 | 0.0482810539158971
optimization step | 30 | -5.905547820039329 | 0.07785427636213159
optimization step | 31 | -5.913249407521742 | 0.0627606285078572
optimization step | 32 | -6.229246238107623 | 0.057348116454766146
optimization step | 33 | -6.233433757815524 | 0.03043418908177174
optimization step | 34 | -6.165361499564318 | 0.0574175755454837
optimization step | 35 | -6.282660777553764 | 0.061092931292840096
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0005,
        "n_blocks": 5,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 5,
        "n_optimization_steps": 18,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+1)
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -6.935187590492762 | 0.033328903589959706
optimization step | 1 | -6.990347375349276 | 0.024595307623949552
optimization step | 2 | -7.095139921064923 | 0.021852509829995283
optimization step | 3 | -7.097936724587925 | 0.027420454968850694
optimization step | 4 | -7.248900202127419 | 0.023304623987580986
optimization step | 5 | -7.408550180087694 | 0.023853402309106892
optimization step | 6 | -7.479210071721264 | 0.011426949674719343
optimization step | 7 | -7.377494250792333 | 0.03488039937316849
optimization step | 8 | -7.549762601196176 | 0.012311703379002585
optimization step | 9 | -7.569322890876217 | 0.0360493061948865
optimization step | 10 | -7.558087202391486 | 0.01358105227396086
optimization step | 11 | -7.634484325386697 | 0.03226519676822079
optimization step | 12 | -7.7036160640717 | 0.02243987631820246
optimization step | 13 | -7.758376911361106 | 0.029119303189841413
optimization step | 14 | -7.697629396764107 | 0.0183296118799316
optimization step | 15 | -7.724459387537593 | 0.05806455923182684
optimization step | 16 | -7.704777788760933 | 0.023783954753760207
optimization step | 17 | -7.724690225505905 | 0.03018020054127031
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0002,
        "n_blocks": 5,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 5,
        "n_optimization_steps": 20,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+1)
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -7.8240871792547875 | 0.013544644260426378
optimization step | 1 | -7.805943260186216 | 0.014563893901524759
optimization step | 2 | -7.819005077638375 | 0.008469257030925733
optimization step | 3 | -7.748604893581981 | 0.02315113196920851
optimization step | 4 | -7.761722197536699 | 0.02155985816175449
optimization step | 5 | -7.874958754339849 | 0.01819251945154451
optimization step | 6 | -7.887306752014713 | 0.018906286738362765
optimization step | 7 | -7.77016169478307 | 0.022772410341530488
optimization step | 8 | -7.879432402176178 | 0.00982954882367169
optimization step | 9 | -7.833281634637823 | 0.03658191019407971
optimization step | 10 | -7.831690761595382 | 0.013894738899532367
optimization step | 11 | -7.848788039927747 | 0.021774852083335152
optimization step | 12 | -7.885352931238556 | 0.019351635532586797
optimization step | 13 | -7.91029855111046 | 0.016520876700000936
optimization step | 14 | -7.859214715374979 | 0.019346488924046765
optimization step | 15 | -7.893854538649718 | 0.048119176370535734
optimization step | 16 | -7.870944942371466 | 0.015672594864976297
optimization step | 17 | -7.8656150264131055 | 0.027910708034106623
optimization step | 18 | -7.931315706110323 | 0.016348761280427164
optimization step | 19 | -7.894056870078271 | 0.02454718331710577
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0002,
        "n_blocks": 10,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 10,
        "n_optimization_steps": 500,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+1)
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -7.912662148705408 | 0.017248727157894096
optimization step | 1 | -7.92231253573402 | 0.015940648116593684
optimization step | 2 | -7.934372197107125 | 0.01742840545436242
optimization step | 3 | -7.950495459809714 | 0.022172640967087306
optimization step | 4 | -7.93834111134961 | 0.016059538435139787
optimization step | 5 | -7.967609463552554 | 0.019190443468067964
optimization step | 6 | -7.978911279046018 | 0.015220332478004757
optimization step | 7 | -7.957599093668622 | 0.013262393132929064
optimization step | 8 | -7.988280054834019 | 0.017625099985287582
optimization step | 9 | -7.957532306988094 | 0.021295856433564568
optimization step | 10 | -8.001365089973506 | 0.022550407384445227
optimization step | 11 | -7.992975809131987 | 0.02285129031204546
optimization step | 12 | -8.005127892768476 | 0.016479415023789096
optimization step | 13 | -7.994917784004407 | 0.025009097334591265
optimization step | 14 | -7.995971577518489 | 0.016732842316769
optimization step | 15 | -8.004677745540874 | 0.012867998003289817
optimization step | 16 | -8.063442527123307 | 0.010789646378483004
optimization step | 17 | -8.009438808930689 | 0.022667650699531295
optimization step | 18 | -8.039382766588213 | 0.013608038811465472
optimization step | 19 | -8.033364223733258 | 0.00991913352666257
optimization step | 20 | -8.004754958613516 | 0.012684381469698244
optimization step | 21 | -8.035903716718561 | 0.015171982381742125
optimization step | 22 | -8.062549928335553 | 0.021619402534311383
optimization step | 23 | -8.080208761348603 | 0.011343901160151876
optimization step | 24 | -8.079014163509807 | 0.01769963743589946
optimization step | 25 | -8.065985320407403 | 0.015756481409683998
optimization step | 26 | -8.064693059143995 | 0.013269988382350845
optimization step | 27 | -8.073937400506392 | 0.0109293490188282
optimization step | 28 | -8.100608063608222 | 0.0186949236525501
optimization step | 29 | -8.054679752840523 | 0.015766055637487952
optimization step | 30 | -8.112958500226409 | 0.01839554629386778
optimization step | 31 | -8.09798492566994 | 0.011695157877170098
optimization step | 32 | -8.064693522551263 | 0.016434396900411038
optimization step | 33 | -8.120870637851079 | 0.018362761713832602
optimization step | 34 | -8.082836951715697 | 0.015568886997207425
optimization step | 35 | -8.08929266385987 | 0.01560777058344512
optimization step | 36 | -8.080061592248267 | 0.01201405682479607
optimization step | 37 | -8.118648934867439 | 0.024414013957852965
optimization step | 38 | -7.996675757043919 | 0.021361219034238554
optimization step | 39 | -7.806895503558936 | 0.024190909067333417
optimization step | 40 | -7.757480287358478 | 0.029817248457622347
optimization step | 41 | -7.67362564415575 | 0.05674278427873764
optimization step | 42 | -7.652720504861046 | 0.014439658459496303
optimization step | 43 | -7.760869867783384 | 0.04491527697573512
optimization step | 44 | -7.278053417738586 | 0.03807612012526662
optimization step | 45 | -5.489321362162445 | 0.06761400446437744
optimization step | 46 | -5.269099392414457 | 0.07701413987471921
optimization step | 47 | 1.249947379743064 | 0.07937826403612332
optimization step | 48 | 4.499424404153122 | 0.15456763130518153
optimization step | 49 | -3.4814027794015674 | 0.15632481949051055
optimization step | 50 | -6.6499361519279265 | 0.05060117033413031
optimization step | 51 | -7.026150319233456 | 0.0809659172146472
optimization step | 52 | -7.457554426048425 | 0.06149324308603686
optimization step | 53 | -7.622824815199616 | 0.05109261229108545
optimization step | 54 | -7.930248204415372 | 0.03533018966937479
optimization step | 55 | -7.858982104835104 | 0.028342275254685766
optimization step | 56 | -7.9664989591891295 | 0.029092506129186308
optimization step | 57 | -8.005555843334587 | 0.03434432969676733
optimization step | 58 | -8.012920294116034 | 0.03820462387706913
optimization step | 59 | -8.092039931411403 | 0.03550584632589354
optimization step | 60 | -8.075277992075238 | 0.04341171666966766
optimization step | 61 | -8.087978141758922 | 0.041330873703682096
optimization step | 62 | -8.129989680162632 | 0.021516324541005168
optimization step | 63 | -8.12950782875529 | 0.0320747817554141
optimization step | 64 | -8.148848361948577 | 0.031116018107469313
optimization step | 65 | -8.203336253698847 | 0.025598898650954844
optimization step | 66 | -8.186749444133886 | 0.020463609716359007
optimization step | 67 | -8.194460258615576 | 0.020988725996686428
optimization step | 68 | -8.191100593334083 | 0.01956725615912652
optimization step | 69 | -8.24702584021426 | 0.011169728215246836
optimization step | 70 | -8.237549441024338 | 0.017200073955561817
optimization step | 71 | -8.237991405433572 | 0.027151538996004115
optimization step | 72 | -8.215070878815299 | 0.02664053435186323
optimization step | 73 | -8.23913045183313 | 0.02356279328611567
optimization step | 74 | -8.240435003060892 | 0.023359888464631797
optimization step | 75 | -8.271543821249148 | 0.018453963776184874
optimization step | 76 | -8.260886061672988 | 0.027310191368796734
optimization step | 77 | -8.258201870230307 | 0.0210871649907367
optimization step | 78 | -8.255706070540755 | 0.018211208446766596
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0007,
        "n_blocks": 5,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 5,
        "n_optimization_steps": 500,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+1)
## Wave Function Parameters
creating wave function parameters file: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | 0.6494319591486027 | 0.32092829966538144
optimization step | 1 | -0.21118200070486898 | 0.09146753687970348
optimization step | 2 | -1.6883570234869112 | 0.06805117684437019
optimization step | 3 | -2.2058133375108597 | 0.03670580339919678
optimization step | 4 | -2.530793876281117 | 0.14025424294697852
optimization step | 5 | -3.1825691451751057 | 0.09625186498400462
optimization step | 6 | -3.4713540344282108 | 0.05178769222396121
optimization step | 7 | -3.8200590268728023 | 0.10271789338175381
optimization step | 8 | -4.13289643113012 | 0.06620911421755123
optimization step | 9 | -4.346526681010452 | 0.11173115075141342
optimization step | 10 | -4.654017647079738 | 0.06073295602853462
optimization step | 11 | -4.809710388816666 | 0.1135825975305297
optimization step | 12 | -5.055715024494492 | 0.04161528228646191
optimization step | 13 | -5.428813303957452 | 0.07803035688408483
optimization step | 14 | -5.383562264604468 | 0.07059422022647008
optimization step | 15 | -5.57974948177354 | 0.04880640809207988
optimization step | 16 | -5.734136487209769 | 0.05179106705301854
optimization step | 17 | -6.017819065330093 | 0.05685844526074957
optimization step | 18 | -6.063474951054355 | 0.040786172147886685
optimization step | 19 | -6.229950287056248 | 0.047981006963249306
optimization step | 20 | -6.204259467310572 | 0.0630629681643118
optimization step | 21 | -6.364725619983151 | 0.03887471661213932
optimization step | 22 | -6.6483166041205575 | 0.06208854063915044
optimization step | 23 | -6.716753716480471 | 0.0603421085632939
optimization step | 24 | -6.696717750397143 | 0.041092854657528835
optimization step | 25 | -6.873251854258018 | 0.03346631101613083
optimization step | 26 | -6.964658143539697 | 0.049000903238216244
optimization step | 27 | -6.966146015330567 | 0.007978388383441818
optimization step | 28 | -7.018737045655763 | 0.03971833476237567
optimization step | 29 | -7.1372439038027835 | 0.030291061333562762
optimization step | 30 | -7.240310839925668 | 0.06062251480289807
optimization step | 31 | -7.138350306727679 | 0.047793954184972634
optimization step | 32 | -7.409949846879101 | 0.045446101658521455
optimization step | 33 | -7.406563589440212 | 0.023189297817470745
optimization step | 34 | -7.378550078509707 | 0.014303053305758331
optimization step | 35 | -7.5019198635994915 | 0.020530515398856784
optimization step | 36 | -7.560515767719977 | 0.016268252429483356
optimization step | 37 | -7.610952363070727 | 0.05225635388108349
optimization step | 38 | -7.854780303558658 | 0.039800290056635594
optimization step | 39 | -7.917611725667939 | 0.03487791197069135
optimization step | 40 | -8.072946535461714 | 0.03924445852429892
optimization step | 41 | -8.050347647297746 | 0.0206455564595162
optimization step | 42 | -8.125621012724475 | 0.03936462438436724
optimization step | 43 | -8.17019427524178 | 0.021451733179152884
optimization step | 44 | -8.26516749101447 | 0.010838828003263532
optimization step | 45 | -8.135868738703369 | 0.028134777323470753
optimization step | 46 | -7.29203698758941 | 0.09463083252455663
optimization step | 47 | -6.331480716048754 | 0.1054351946501218
optimization step | 48 | 2.80082301957454 | 0.16612011949600425
optimization step | 49 | -6.194426393927844 | 0.11146075837397408
optimization step | 50 | -1.797603717866878 | 0.13812943381003784
optimization step | 51 | -7.092464064060867 | 0.0721953345649369
optimization step | 52 | -7.571627181668974 | 0.11565919244142725
optimization step | 53 | -6.702638210847303 | 0.1003622577977233
optimization step | 54 | -3.1089387330079354 | 0.12991536260849454
optimization step | 55 | -2.6677977434401186 | 0.15858245929142506
optimization step | 56 | -3.453908087048086 | 0.1393602498246674
optimization step | 57 | -4.224076929985361 | 0.0852568320829063
optimization step | 58 | -5.2609540128493215 | 0.07448582606862214
optimization step | 59 | -5.907615545684426 | 0.09597712720952051
optimization step | 60 | -6.113566221724295 | 0.07644960152733125
optimization step | 61 | -6.247088697408074 | 0.08905201789800204
optimization step | 62 | -6.320940600774917 | 0.07492963599963515
optimization step | 63 | -6.518911391131896 | 0.04636839098185498
optimization step | 64 | -6.582653914194435 | 0.05007037467200411
optimization step | 65 | -6.703750812942738 | 0.0551585199151282
optimization step | 66 | -6.7171514801142775 | 0.026953604489837378
optimization step | 67 | -6.8735307096749825 | 0.05712778659394072
optimization step | 68 | -6.945530863824368 | 0.024478272446133733
optimization step | 69 | -6.98173950132059 | 0.048270821349789626
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.001,
        "n_blocks": 5,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 5,
        "n_optimization_steps": 500,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma",
            "tau",
            "sigma_tau"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+tau+sigma_tau+1)
## Wave Function Parameters
creating wave function parameters file: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | 0.187179768499823 | 0.5248844765556363
optimization step | 1 | 0.23526015817396928 | 0.1902520118239307
optimization step | 2 | -1.683178863967079 | 0.09346551060199638
optimization step | 3 | -0.0576941542828802 | 0.2752803340765705
optimization step | 4 | 0.5854982485604098 | 0.5187345727258933
optimization step | 5 | -2.7921360240754787 | 0.11030140934178886
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0002,
        "n_blocks": 5,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 5,
        "n_optimization_steps": 500,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma",
            "tau",
            "sigma_tau"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+tau+sigma_tau+1)
## Wave Function Parameters
creating wave function parameters file: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | 0.187179768499823 | 0.5248844765556363
optimization step | 1 | -0.086441341384874 | 0.20611287502093467
optimization step | 2 | -0.3141938227399096 | 0.22584433866958767
optimization step | 3 | -1.1750897919220147 | 0.20114322387950812
optimization step | 4 | -1.1885448327210035 | 0.17777585053585432
optimization step | 5 | -1.7231996561543874 | 0.12691490770814184
optimization step | 6 | -1.746806595768431 | 0.17061691234592968
optimization step | 7 | -2.062028697550861 | 0.07616009853044912
optimization step | 8 | -2.443847342467787 | 0.10269136539869952
optimization step | 9 | -2.4688544630292375 | 0.1361550857516854
optimization step | 10 | -2.5255213425698746 | 0.09929166674829455
optimization step | 11 | -2.7458702463483426 | 0.17127336304644988
optimization step | 12 | -2.965449019884074 | 0.09990860508310238
optimization step | 13 | -3.054216673116888 | 0.13616617384719384
optimization step | 14 | -3.1438519279876203 | 0.09851672833771402
optimization step | 15 | -4.161077538329249 | 0.7604747215815528
optimization step | 16 | -3.410600334234991 | 0.10357600906249782
optimization step | 17 | -3.6393624285826016 | 0.05689064508776535
optimization step | 18 | -3.5870478965762134 | 0.060895807715743225
optimization step | 19 | -3.780302280320039 | 0.09556228338939819
optimization step | 20 | -3.5691522048594253 | 0.022224959126306783
optimization step | 21 | -3.9471187712349156 | 0.09027228966857878
optimization step | 22 | -3.953929957709429 | 0.07787075939856536
optimization step | 23 | -4.1331813767059575 | 0.13629067420216096
optimization step | 24 | -3.900616138698858 | 0.09918387104269308
optimization step | 25 | -4.300788869518616 | 0.035412755981781854
optimization step | 26 | -4.319917892877745 | 0.11421792677479527
optimization step | 27 | -4.386890791382295 | 0.027413501549731493
optimization step | 28 | -4.519637628274139 | 0.0816831679260325
optimization step | 29 | -4.6935720368769305 | 0.12174072174180592
optimization step | 30 | -4.769129043185648 | 0.12460879556421531
optimization step | 31 | -4.5944390774319706 | 0.07349143147094822
optimization step | 32 | -4.959159044201938 | 0.06214984228200563
optimization step | 33 | -4.85640639369593 | 0.06642364094460869
optimization step | 34 | -4.83312594637958 | 0.06930499603618691
optimization step | 35 | -4.958841154459848 | 0.08094954426012516
optimization step | 36 | -5.051441051439543 | 0.08969887320384475
optimization step | 37 | -5.076779131540865 | 0.10186415866003992
optimization step | 38 | -5.021564917316392 | 0.03179968285647934
optimization step | 39 | -5.038902214647919 | 0.066187265089746
optimization step | 40 | -5.345557110387253 | 0.07343641492431041
optimization step | 41 | -5.390298065063394 | 0.024298604122055725
optimization step | 42 | -5.413291080610085 | 0.12320947383746252
optimization step | 43 | -5.364540893052678 | 0.09816680980379447
optimization step | 44 | -5.420380392701256 | 0.08112905322511178
optimization step | 45 | -5.642715218003177 | 0.08283261701715645
optimization step | 46 | -5.598685721764764 | 0.025215748298076112
optimization step | 47 | -5.785178320962453 | 0.09532097424387241
optimization step | 48 | -5.7876313991686095 | 0.1265880675285258
optimization step | 49 | -5.741957733316115 | 0.08700798391141437
optimization step | 50 | -5.808617709660998 | 0.11477919091771652
optimization step | 51 | -5.915455175326534 | 0.05019923721181174
optimization step | 52 | -5.856824005225606 | 0.06539500097920381
optimization step | 53 | -5.861200740925502 | 0.11206607302706144
optimization step | 54 | -5.956723630847331 | 0.08866244200530504
optimization step | 55 | -5.980392537965321 | 0.05039772429264021
optimization step | 56 | -5.843054048892946 | 0.051768285372179326
optimization step | 57 | -6.062809988607664 | 0.04055487097303537
optimization step | 58 | -6.069266966304011 | 0.03573302808416098
optimization step | 59 | -6.164039445846639 | 0.07056394764169333
optimization step | 60 | -6.270680865989726 | 0.06348063092984238
optimization step | 61 | -6.185074415547137 | 0.07955940493751258
optimization step | 62 | -6.2527923006775366 | 0.030120742148872878
optimization step | 63 | -6.387543083316293 | 0.042101748718829525
optimization step | 64 | -6.366193606648389 | 0.04239144114804545
optimization step | 65 | -6.379952168032387 | 0.04946074195508474
optimization step | 66 | -6.421017687902389 | 0.040951704639595674
optimization step | 67 | -6.428518047416276 | 0.06942889234541202
optimization step | 68 | -6.373161585148862 | 0.10444971111198459
optimization step | 69 | -6.482999681350771 | 0.09409585357832244
optimization step | 70 | -6.483767038399736 | 0.031060311371266046
optimization step | 71 | -6.410423273053459 | 0.07475235266623785
optimization step | 72 | -6.462947361353102 | 0.03911165161387399
optimization step | 73 | -6.583974631853192 | 0.06792155681371112
optimization step | 74 | -6.760693601230264 | 0.02927505070702041
optimization step | 75 | -6.532618384587153 | 0.03372102623168739
optimization step | 76 | -6.649686510205514 | 0.04717079953944696
optimization step | 77 | -6.617326057577872 | 0.04893112597137913
optimization step | 78 | -6.6945345423664175 | 0.027004958390833052
optimization step | 79 | -6.810045116046913 | 0.021406501559502595
optimization step | 80 | -6.717704299856726 | 0.044684515106253514
optimization step | 81 | -6.797398004645283 | 0.06352021979193206
optimization step | 82 | -6.726886062782455 | 0.05390916767535227
optimization step | 83 | -6.7570165567804255 | 0.05787963072821936
optimization step | 84 | -6.8535244543606835 | 0.044248525049992465
optimization step | 85 | -6.921282644868597 | 0.05017734396524803
optimization step | 86 | -6.999567613058008 | 0.037646229124943295
optimization step | 87 | -6.873848909734727 | 0.07037401503965245
optimization step | 88 | -6.9028837140464985 | 0.03432562193251085
optimization step | 89 | -6.919900486454258 | 0.03577559054413635
optimization step | 90 | -6.977656085142426 | 0.06493457850670487
optimization step | 91 | -6.9729205548227595 | 0.023558522527121387
optimization step | 92 | -7.097681693969872 | 0.027087401562123917
optimization step | 93 | -6.98314333585337 | 0.026195635633508022
optimization step | 94 | -7.014829788806185 | 0.04414789907728011
optimization step | 95 | -7.039643315768396 | 0.03228561558750363
optimization step | 96 | -7.062098567056434 | 0.05230322804456293
optimization step | 97 | -7.103222018987718 | 0.02822191359358623
optimization step | 98 | -7.205123947521369 | 0.061225587799835855
optimization step | 99 | -7.192586387497501 | 0.04620226890292976
optimization step | 100 | -7.159549501819593 | 0.04315608562291989
optimization step | 101 | -7.175571153353658 | 0.025744672835593993
optimization step | 102 | -7.215529140845916 | 0.03997024987319631
optimization step | 103 | -7.297230190698239 | 0.046336286937342874
optimization step | 104 | -7.169110596004924 | 0.039951654321188024
optimization step | 105 | -7.252774225777081 | 0.025150020423929383
optimization step | 106 | -7.3388498760786405 | 0.016635649896527376
optimization step | 107 | -7.295754489184847 | 0.04196063821002091
optimization step | 108 | -7.38512963567958 | 0.03417215989226934
optimization step | 109 | -7.276383748181186 | 0.04568186249719059
optimization step | 110 | -7.364668573723153 | 0.018528999968478045
optimization step | 111 | -7.387591027641292 | 0.030398761755132134
optimization step | 112 | -7.37957622504307 | 0.03347406768212661
optimization step | 113 | -7.4162058220825084 | 0.022930762907510453
optimization step | 114 | -7.444243208018162 | 0.043304416151479654
optimization step | 115 | -7.450220794353579 | 0.04476157402371325
optimization step | 116 | -7.4175396751150755 | 0.0566034109916402
optimization step | 117 | -7.458656213644192 | 0.04538588496530056
optimization step | 118 | -7.555697366467077 | 0.05872256345416363
optimization step | 119 | -7.5341075985959405 | 0.022296888629369942
optimization step | 120 | -7.462503105177211 | 0.031905845218069454
optimization step | 121 | -7.538852583113794 | 0.042677144040160056
optimization step | 122 | -7.563088669711311 | 0.0340979815083253
optimization step | 123 | -7.608460692782801 | 0.02972181141478294
optimization step | 124 | -7.54134058854095 | 0.04775761127581212
optimization step | 125 | -7.542422620400513 | 0.014852255452266278
optimization step | 126 | -7.553483800498547 | 0.048762816282172876
optimization step | 127 | -7.583913311591807 | 0.02039440168951125
optimization step | 128 | -7.558545369912518 | 0.03923683647675511
optimization step | 129 | -7.569230051371784 | 0.04173093438904378
optimization step | 130 | -7.576791217611984 | 0.028697526545508837
optimization step | 131 | -7.546823398504247 | 0.055660319641836375
optimization step | 132 | -7.59401480114294 | 0.036001790641031324
optimization step | 133 | -7.666930580066553 | 0.045789929873967315
optimization step | 134 | -7.575243423977781 | 0.06110760781269718
optimization step | 135 | -7.790511483547941 | 0.02887951676845865
optimization step | 136 | -7.73412285579725 | 0.033028002295894136
optimization step | 137 | -7.794102183437414 | 0.026914676651021333
optimization step | 138 | -7.768153572097127 | 0.014940533128897598
optimization step | 139 | -7.737477674775772 | 0.027432030548667802
optimization step | 140 | -7.788912463763599 | 0.029039749451319225
optimization step | 141 | -7.747300680954092 | 0.03658183306889225
optimization step | 142 | -7.745739756162498 | 0.01549863749008281
optimization step | 143 | -7.890996432267668 | 0.0566684537155536
optimization step | 144 | -7.756791443583587 | 0.028268876032481752
optimization step | 145 | -7.807816357052562 | 0.02236660650719675
optimization step | 146 | -7.894962763408193 | 0.04586431484297789
optimization step | 147 | -7.854201008335286 | 0.03183295543007872
optimization step | 148 | -7.85154456092994 | 0.02836226415561435
optimization step | 149 | -7.891715498365725 | 0.02393959399746391
optimization step | 150 | -7.8477938884101395 | 0.019097400395204637
optimization step | 151 | -7.860687171482169 | 0.0366022863304869
optimization step | 152 | -7.905968546995811 | 0.03405831140128101
optimization step | 153 | -7.925298190968445 | 0.015603356906654257
optimization step | 154 | -7.92432538753463 | 0.03309464284670079
optimization step | 155 | -7.898977363805399 | 0.019238614225853712
optimization step | 156 | -7.950136608379888 | 0.018521223949073823
optimization step | 157 | -7.884986720496656 | 0.020294399616478602
optimization step | 158 | -7.955506810547639 | 0.01597060647722909
optimization step | 159 | -7.936305714118026 | 0.02091794377634795
optimization step | 160 | -7.9083333302441785 | 0.015623408531603808
optimization step | 161 | -7.930581943625819 | 0.01548410636456299
optimization step | 162 | -7.981211570314812 | 0.035424479551027024
optimization step | 163 | -7.986515326801 | 0.009705550103956977
optimization step | 164 | -7.9450331319821075 | 0.020594694460476418
optimization step | 165 | -7.940683536418518 | 0.036178493661760104
optimization step | 166 | -7.984514490094003 | 0.007555555516902494
optimization step | 167 | -7.989136263995038 | 0.011673823068578762
optimization step | 168 | -8.015595690836369 | 0.0229825428950705
optimization step | 169 | -8.042572111916115 | 0.014332455602449033
optimization step | 170 | -8.05708599309858 | 0.01902122111183997
optimization step | 171 | -8.034673711770937 | 0.01920764815649769
optimization step | 172 | -8.061756367835635 | 0.02357692710339003
optimization step | 173 | -8.030722044238143 | 0.02381640616738665
optimization step | 174 | -8.030521800046612 | 0.028977072497420205
optimization step | 175 | -8.02594151019577 | 0.01638335550055291
optimization step | 176 | -8.001381589519502 | 0.026509873704726513
optimization step | 177 | -8.106078032608531 | 0.010403355811817193
optimization step | 178 | -8.09340618448205 | 0.025558269889161187
optimization step | 179 | -8.109252205771167 | 0.024085799144560657
optimization step | 180 | -8.138914675833414 | 0.018356319413408147
optimization step | 181 | -8.085286783298738 | 0.014422839327707599
optimization step | 182 | -8.138388862375825 | 0.024091730718473356
optimization step | 183 | -8.07423623837222 | 0.024960523127646104
optimization step | 184 | -8.100747499327042 | 0.027432894044601263
optimization step | 185 | -8.09410043877365 | 0.01916906206018173
optimization step | 186 | -8.141036114060972 | 0.029021500144824363
optimization step | 187 | -8.14520055163119 | 0.020263205987698197
optimization step | 188 | -8.112095706591557 | 0.028585970108591326
optimization step | 189 | -8.17238085890253 | 0.009778977519867643
optimization step | 190 | -8.142478949683206 | 0.019715776300813365
optimization step | 191 | -8.180445395993337 | 0.021461946371568525
optimization step | 192 | -8.150205472813001 | 0.012489434574551254
optimization step | 193 | -8.186719432728074 | 0.02325933026243425
optimization step | 194 | -8.129569327703852 | 0.02858723510685812
optimization step | 195 | -8.16900961834135 | 0.0064766530141777134
optimization step | 196 | -8.170965575393804 | 0.003524418955749163
optimization step | 197 | -8.190375113885445 | 0.026799346168158898
optimization step | 198 | -8.1923675938196 | 0.032454963339383934
optimization step | 199 | -8.186140124428771 | 0.023300888945850452
optimization step | 200 | -8.155871682222664 | 0.008526833532132272
optimization step | 201 | -8.181373318035117 | 0.020984747560755457
optimization step | 202 | -8.196607835183562 | 0.013110872597117672
optimization step | 203 | -8.200772248252985 | 0.028683592111525682
optimization step | 204 | -8.198536314922167 | 0.01673415841293247
optimization step | 205 | -8.196223987472798 | 0.011892363165343468
optimization step | 206 | -8.190027108509268 | 0.017388146126304124
optimization step | 207 | -8.25626113656644 | 0.02319637065704894
optimization step | 208 | -8.219540938283437 | 0.00803046529104358
optimization step | 209 | -8.221630268342901 | 0.027810517470155716
optimization step | 210 | -8.18590583593555 | 0.01781520179416719
optimization step | 211 | -8.180015516464191 | 0.023006401532677783
optimization step | 212 | -8.232920585238713 | 0.025932188548654537
optimization step | 213 | -8.21271280577128 | 0.02930452078075812
optimization step | 214 | -8.259537074750837 | 0.014729290592125951
optimization step | 215 | -8.240053929064278 | 0.019565661845341788
optimization step | 216 | -8.263189559476777 | 0.026972198030726677
optimization step | 217 | -8.240123608074068 | 0.03210000019763953
optimization step | 218 | -8.250299491487679 | 0.01941744736115211
optimization step | 219 | -8.230317899800152 | 0.012307505394381666
optimization step | 220 | -8.261284731564078 | 0.023612833424018514
optimization step | 221 | -8.22196490052384 | 0.012701392802777564
optimization step | 222 | -8.271421729615515 | 0.026954655466587485
optimization step | 223 | -8.253638133309146 | 0.01431742771261628
optimization step | 224 | -8.206502714481676 | 0.0055738330830702055
optimization step | 225 | -8.254412107805347 | 0.024931383305585628
optimization step | 226 | -8.294613812788516 | 0.018718923876398657
optimization step | 227 | -8.254927661462213 | 0.02879797163365831
optimization step | 228 | -8.269346634377067 | 0.021722315579725343
optimization step | 229 | -8.228563439455035 | 0.018662799125513788
optimization step | 230 | -8.230829278024158 | 0.031096633239589874
optimization step | 231 | -8.264906680668465 | 0.011897693014676669
optimization step | 232 | -8.300452633592597 | 0.015216616603920274
optimization step | 233 | -8.281695606117113 | 0.01140733003714949
optimization step | 234 | -8.279646121142644 | 0.014516886900450842
optimization step | 235 | -8.325521437623026 | 0.014344343806436028
optimization step | 236 | -8.276223876002485 | 0.014012867533337362
optimization step | 237 | -8.265566913390442 | 0.017017055783367735
optimization step | 238 | -8.281173678568923 | 0.017982706626282104
optimization step | 239 | -8.30327928745495 | 0.01050322848980102
optimization step | 240 | -8.280857208605315 | 0.01299945639941284
optimization step | 241 | -8.30250606166853 | 0.01872809965516059
optimization step | 242 | -8.33789850911589 | 0.0228193247593891
optimization step | 243 | -8.287279504149623 | 0.023513431276660172
optimization step | 244 | -8.28865460134639 | 0.0065925629166233655
optimization step | 245 | -8.339302070521255 | 0.01577347669278463
optimization step | 246 | -8.320606195659131 | 0.01999894253042076
optimization step | 247 | -8.319298113609777 | 0.00789331094836164
optimization step | 248 | -8.30470390660134 | 0.021235165671776347
optimization step | 249 | -8.322842483281352 | 0.010359697300468054
optimization step | 250 | -8.309519765238571 | 0.017211571221692375
optimization step | 251 | -8.346795289082516 | 0.021356343784199098
optimization step | 252 | -8.327777552308099 | 0.01499422664542845
optimization step | 253 | -8.308552612127313 | 0.01966204593752063
optimization step | 254 | -8.350174984378274 | 0.01044601778800049
optimization step | 255 | -8.329220070287588 | 0.01390027663956066
optimization step | 256 | -8.319235743940718 | 0.014244400688835855
optimization step | 257 | -8.342918480872886 | 0.01367477018813975
optimization step | 258 | -8.337034043696821 | 0.011553655274873299
optimization step | 259 | -8.301556746619118 | 0.008107915333298342
optimization step | 260 | -8.32686027617153 | 0.012761471532865859
optimization step | 261 | -8.343110511091197 | 0.01687918144698412
optimization step | 262 | -8.354882985427007 | 0.014878584763061944
optimization step | 263 | -8.3203527320457 | 0.01761657516568708
optimization step | 264 | -8.360751636513479 | 0.017756935210065793
optimization step | 265 | -8.34208806571051 | 0.010230086471406327
optimization step | 266 | -8.32821866484373 | 0.009505637244200251
optimization step | 267 | -8.379913264592172 | 0.00987896857696669
optimization step | 268 | -8.341211719907 | 0.009428199074873987
optimization step | 269 | -8.341951617118855 | 0.017705411917874503
optimization step | 270 | -8.361854757630987 | 0.006206953496958219
optimization step | 271 | -8.338974090045095 | 0.018940774038232546
optimization step | 272 | -8.391825717621433 | 0.008518490362746914
optimization step | 273 | -8.362123874726986 | 0.010326547958534424
optimization step | 274 | -8.364743078381867 | 0.019811960307893145
optimization step | 275 | -8.359170299212064 | 0.011496361528768453
optimization step | 276 | -8.370949122942504 | 0.025027943371815336
optimization step | 277 | -8.37264910412693 | 0.012646765095478546
optimization step | 278 | -8.410307110566245 | 0.009634583340446042
optimization step | 279 | -8.411608918711323 | 0.011093710776351438
optimization step | 280 | -8.35080100495249 | 0.01342515991762627
optimization step | 281 | -8.383636812415311 | 0.016168798347587804
optimization step | 282 | -8.375412981421853 | 0.017329595611599382
optimization step | 283 | -8.377902674496449 | 0.012124099160267411
optimization step | 284 | -8.401816941353042 | 0.014723782117389329
optimization step | 285 | -8.377514698184957 | 0.016043748409117632
optimization step | 286 | -8.388665711155642 | 0.011461486024510891
optimization step | 287 | -8.38898495145524 | 0.017853118589738003
optimization step | 288 | -8.37609239208094 | 0.016767374778467344
optimization step | 289 | -8.3894642822845 | 0.0104121061104938
optimization step | 290 | -8.393621934592058 | 0.011809179948891416
optimization step | 291 | -8.401601465265044 | 0.00680642616948845
optimization step | 292 | -8.386099199839462 | 0.013878816031006151
optimization step | 293 | -8.376227376226018 | 0.014590282061570118
optimization step | 294 | -8.403021870988644 | 0.009622025921967764
optimization step | 295 | -8.364134127735806 | 0.004589841874562653
optimization step | 296 | -8.396944575149082 | 0.004069117494517048
optimization step | 297 | -8.38707009093832 | 0.019976550931245264
optimization step | 298 | -8.403009512097466 | 0.017853287047908846
optimization step | 299 | -8.403835191424587 | 0.012632574106464832
optimization step | 300 | -8.417326454141522 | 0.01277060163848717
optimization step | 301 | -8.42198141737392 | 0.00868993629046976
optimization step | 302 | -8.405123766159251 | 0.014453580416583891
optimization step | 303 | -8.424271081768675 | 0.010683190234251499
optimization step | 304 | -8.400255771514855 | 0.008080070494535274
optimization step | 305 | -8.430696201837458 | 0.025381896908459753
optimization step | 306 | -8.426621515141296 | 0.0168049983245617
optimization step | 307 | -8.418695020309352 | 0.017293223524882946
optimization step | 308 | -8.425055436329327 | 0.005350003585845072
optimization step | 309 | -8.413483403761214 | 0.013090596512170554
optimization step | 310 | -8.428515569056376 | 0.010122172642897519
optimization step | 311 | -8.42668193076928 | 0.015661110216225665
optimization step | 312 | -8.404276384856212 | 0.004473953006894554
optimization step | 313 | -8.391308288629592 | 0.008526444712320003
optimization step | 314 | -8.425444500302437 | 0.008494836576649943
optimization step | 315 | -8.434066298927537 | 0.014110264845517771
optimization step | 316 | -8.415312038777737 | 0.014323914150881065
optimization step | 317 | -8.437236979295257 | 0.005969275614803367
optimization step | 318 | -8.41849937768635 | 0.010413940109218941
optimization step | 319 | -8.426950296302385 | 0.010892935343134322
optimization step | 320 | -8.428764706450062 | 0.022233855183288697
optimization step | 321 | -8.429880679231713 | 0.0050748144258234325
optimization step | 322 | -8.408433427357082 | 0.00529377021199593
optimization step | 323 | -8.418111244891167 | 0.006132989063741653
optimization step | 324 | -8.429525495224354 | 0.016821690627003373
optimization step | 325 | -8.43058870488738 | 0.011940801635706027
optimization step | 326 | -8.420854688664596 | 0.009087594335898274
optimization step | 327 | -8.410079392471724 | 0.009295380832879722
optimization step | 328 | -8.395778655459614 | 0.013270811353382157
optimization step | 329 | -8.418183611222123 | 0.009077208846706313
optimization step | 330 | -8.425326589915871 | 0.013276922562239387
optimization step | 331 | -8.413072946450555 | 0.014386529732158452
optimization step | 332 | -8.428051963036111 | 0.010650830987360232
optimization step | 333 | -8.434767786919549 | 0.016458115453308365
optimization step | 334 | -8.444621583973369 | 0.010933835030973638
optimization step | 335 | -8.439345691334111 | 0.015536110329962426
optimization step | 336 | -8.426571868606272 | 0.010020862784803583
optimization step | 337 | -8.435472275583596 | 0.012188649831330325
optimization step | 338 | -8.433704378847597 | 0.004809700744370103
optimization step | 339 | -8.454048736845843 | 0.011355973862442336
optimization step | 340 | -8.454295937035075 | 0.0046437992293905436
optimization step | 341 | -8.44310362824398 | 0.01344967826532895
optimization step | 342 | -8.440948926455256 | 0.016172478556690528
optimization step | 343 | -8.428283261673235 | 0.011877275296410849
optimization step | 344 | -8.427441577230564 | 0.010267978704172686
optimization step | 345 | -8.441478357569673 | 0.014698463115003096
optimization step | 346 | -8.437102441144699 | 0.011793573716143493
optimization step | 347 | -8.456259104605024 | 0.0070864585695485025
optimization step | 348 | -8.422917365053362 | 0.009915003613687759
optimization step | 349 | -8.456685881174975 | 0.010584192724210548
optimization step | 350 | -8.43551608441545 | 0.01149140494524702
optimization step | 351 | -8.440953005487168 | 0.02062477410188321
optimization step | 352 | -8.43811418635312 | 0.024387512381296193
optimization step | 353 | -8.440479159132789 | 0.006908131619096986
optimization step | 354 | -8.436197053716093 | 0.017302950900027418
optimization step | 355 | -8.420157236593678 | 0.014133490311652671
optimization step | 356 | -8.447545543428387 | 0.005910205992276889
optimization step | 357 | -8.44150223185169 | 0.0062721051102061325
optimization step | 358 | -8.437743076777046 | 0.00866179568787709
optimization step | 359 | -8.442139705620141 | 0.010000942120224739
optimization step | 360 | -8.439521263767983 | 0.017908348121728687
optimization step | 361 | -8.431307514932644 | 0.015010878631877394
optimization step | 362 | -8.450434119158878 | 0.0061937107076708445
optimization step | 363 | -8.438723307750172 | 0.007056054046007443
optimization step | 364 | -8.445656355863713 | 0.015169946789130188
optimization step | 365 | -8.42731011698698 | 0.014339673099820108
optimization step | 366 | -8.438078247412163 | 0.012327858105123284
optimization step | 367 | -8.452232406101619 | 0.009004123821892768
optimization step | 368 | -8.447392301151543 | 0.009267722350814307
optimization step | 369 | -8.451270184544947 | 0.018285796287391968
optimization step | 370 | -8.416686875587402 | 0.009122309279946221
optimization step | 371 | -8.446120047190018 | 0.01323605863658974
optimization step | 372 | -8.44021338380658 | 0.01538746360542365
optimization step | 373 | -8.441290668422997 | 0.01041851793072204
optimization step | 374 | -8.440466254002919 | 0.00930681561888717
optimization step | 375 | -8.436215250992124 | 0.015014981788535543
optimization step | 376 | -8.443455311162687 | 0.0055523612309651875
optimization step | 377 | -8.467236697877068 | 0.007086831065182026
optimization step | 378 | -8.4458763083518 | 0.0068750053768736556
optimization step | 379 | -8.453728497176318 | 0.018747114789779473
optimization step | 380 | -8.429289333616413 | 0.008188116660271747
optimization step | 381 | -8.432438397257268 | 0.013260030871160079
optimization step | 382 | -8.432449129750122 | 0.0046081028963696575
optimization step | 383 | -8.461222265660602 | 0.01629139205199389
optimization step | 384 | -8.442389817839066 | 0.01170942386645056
optimization step | 385 | -8.454833004905216 | 0.01163165752405658
optimization step | 386 | -8.43771696654662 | 0.010041436184895704
optimization step | 387 | -8.433352466240802 | 0.015111096180814075
optimization step | 388 | -8.447768660556779 | 0.019127602235998156
optimization step | 389 | -8.458512393116555 | 0.013148322744483108
optimization step | 390 | -8.433756755489625 | 0.007666316313215764
optimization step | 391 | -8.428448325044553 | 0.009085824487262258
optimization step | 392 | -8.446242746515273 | 0.008451723941594241
optimization step | 393 | -8.470682099666938 | 0.008136993137993058
optimization step | 394 | -8.437837459193856 | 0.013865578560145201
optimization step | 395 | -8.451742571171575 | 0.016665098545504353
optimization step | 396 | -8.443910159377932 | 0.011790993022492167
optimization step | 397 | -8.438479511106646 | 0.014089876421609351
optimization step | 398 | -8.459838930617954 | 0.0066260711365235874
optimization step | 399 | -8.449419320490843 | 0.007872874662313201
optimization step | 400 | -8.426891917714988 | 0.014907254810921778
optimization step | 401 | -8.45292240407035 | 0.008946077322432862
optimization step | 402 | -8.459599246581819 | 0.005817426902016267
optimization step | 403 | -8.46746810457958 | 0.00964163294180199
optimization step | 404 | -8.45391751393297 | 0.004116546657137546
optimization step | 405 | -8.416460648808647 | 0.012061327269782282
optimization step | 406 | -8.45102691090211 | 0.01166773208416747
optimization step | 407 | -8.444844988522817 | 0.0062046416522030185
optimization step | 408 | -8.414954991307049 | 0.014318720106909768
optimization step | 409 | -8.430276831359398 | 0.011014018281208444
optimization step | 410 | -8.439524632973354 | 0.013419996869848625
optimization step | 411 | -8.44798185574125 | 0.00732746477433716
optimization step | 412 | -8.442687216912717 | 0.008551631386704386
optimization step | 413 | -8.456005636403601 | 0.00874325265253301
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0002,
        "n_blocks": 5,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 5,
        "n_optimization_steps": 500,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+1)
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -6.743301953634699 | 0.03638398847325081
optimization step | 1 | -7.141189963566563 | 0.018167449821937017
optimization step | 2 | -6.993504198677703 | 0.11117007044294519
optimization step | 3 | -1.9885931016927327 | 0.3056089413850535
optimization step | 4 | -0.7636837777257545 | 0.07848631975289765
optimization step | 5 | -3.361451365941843 | 0.05525454979066071
optimization step | 6 | -3.2702412015595135 | 0.05585487866057384
optimization step | 7 | -3.462099299961584 | 0.12421848233148967
optimization step | 8 | -3.461628479931563 | 0.09880409723308009
optimization step | 9 | -3.39374445394823 | 0.09974865115683244
optimization step | 10 | -3.4652939217303405 | 0.05183868190705648
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0002,
        "n_blocks": 5,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 5,
        "n_optimization_steps": 500,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+1)
## Wave Function Parameters
creating wave function parameters file: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | 0.6494319591486027 | 0.32092829966538144
optimization step | 1 | 0.3511868892977009 | 0.49166917142011896
optimization step | 2 | -0.4101995212730075 | 0.10800956702773815
optimization step | 3 | -0.6264534811160647 | 0.13365896297195295
optimization step | 4 | -1.2797276246839426 | 0.16296613037421706
optimization step | 5 | -1.5669007235673553 | 0.1610060693487424
optimization step | 6 | -1.735489047837507 | 0.13306378769346064
optimization step | 7 | -2.114072592603093 | 0.16305376512121333
optimization step | 8 | -2.372012605751627 | 0.13526435557968616
optimization step | 9 | -2.2586554501222986 | 0.12879958950064346
optimization step | 10 | -2.5373541990482464 | 0.06980517126931444
optimization step | 11 | -2.5349921642824595 | 0.16469667463756835
optimization step | 12 | -2.758497247213678 | 0.08149898869613266
optimization step | 13 | -2.9720679645182426 | 0.16326730544145465
optimization step | 14 | -3.0456309936807693 | 0.10486073081969395
optimization step | 15 | -3.185511355481727 | 0.04936551710680025
optimization step | 16 | -3.3028093050958076 | 0.07549169329384091
optimization step | 17 | -3.4598799906616904 | 0.07195521032492187
optimization step | 18 | -3.4507188476669315 | 0.08262853016892856
optimization step | 19 | -3.712304242828894 | 0.084258493401331
optimization step | 20 | -3.4170056777429516 | 0.03744659753549304
optimization step | 21 | -3.853966272244926 | 0.0793920019634066
optimization step | 22 | -3.845471644846506 | 0.06810228225400301
optimization step | 23 | -3.9535960425995924 | 0.12911954299572484
optimization step | 24 | -3.7561084508203657 | 0.11890069206267402
optimization step | 25 | -4.164607594637159 | 0.04429777966238013
optimization step | 26 | -4.243367255505984 | 0.11827512122107606
optimization step | 27 | -4.232589160226537 | 0.04078496502263729
optimization step | 28 | -4.324603322357359 | 0.09756134884516876
optimization step | 29 | -4.509431613403719 | 0.12247066112092128
optimization step | 30 | -4.640162830258193 | 0.12390563523476178
optimization step | 31 | -4.439144495269496 | 0.08108505246413705
optimization step | 32 | -4.792760524594027 | 0.08564339395066108
optimization step | 33 | -4.715211423756225 | 0.0694627620191435
optimization step | 34 | -4.743223674487568 | 0.043839104415456256
optimization step | 35 | -4.853073807381791 | 0.09287242263583574
optimization step | 36 | -4.858012596924235 | 0.08248758274175676
optimization step | 37 | -4.979670151196325 | 0.09263670031432318
optimization step | 38 | -4.904639051712115 | 0.026867401959602213
optimization step | 39 | -4.944650655145988 | 0.07684156901113394
optimization step | 40 | -5.182058925579208 | 0.06273597742941697
optimization step | 41 | -5.286287630158431 | 0.04146270522003201
optimization step | 42 | -5.258630219447959 | 0.1213640850923996
optimization step | 43 | -5.307211614488941 | 0.09736465363422984
optimization step | 44 | -5.2881398061570515 | 0.07839289456093938
optimization step | 45 | -5.541288371632998 | 0.08796741273075263
optimization step | 46 | -5.4971937962835415 | 0.06021858973114726
optimization step | 47 | -5.624428234085096 | 0.07093056323666298
optimization step | 48 | -5.64290460801559 | 0.13188026074047413
optimization step | 49 | -5.6676745122489445 | 0.08988008239007228
optimization step | 50 | -5.643876468019512 | 0.14335247942998325
optimization step | 51 | -5.831258240881011 | 0.06684095092192657
optimization step | 52 | -5.724937180369871 | 0.05170530025907112
optimization step | 53 | -5.781199033206702 | 0.10737092979639047
optimization step | 54 | -5.8615456798404235 | 0.07696543577155136
optimization step | 55 | -5.859879098167012 | 0.06221409109525344
optimization step | 56 | -5.739079809869901 | 0.037002366244170415
optimization step | 57 | -5.991555384383414 | 0.049467157877120305
optimization step | 58 | -6.016272887418072 | 0.057418338543682855
optimization step | 59 | -6.069631238412965 | 0.07402727693403989
optimization step | 60 | -6.197913817751254 | 0.05816612539981406
optimization step | 61 | -6.106465504120667 | 0.056280330987763076
optimization step | 62 | -6.180280840994809 | 0.035057599687661804
optimization step | 63 | -6.303969218125625 | 0.04352743940713664
optimization step | 64 | -6.273930110824814 | 0.060235455406850054
optimization step | 65 | -6.297148804160931 | 0.06241386569352338
optimization step | 66 | -6.3370965349617006 | 0.05223538342083204
optimization step | 67 | -6.366604049069184 | 0.057100623549182175
optimization step | 68 | -6.295797881000614 | 0.09863502286262454
optimization step | 69 | -6.438409050607342 | 0.09171800928166166
optimization step | 70 | -6.41177597925452 | 0.026555485077385255
optimization step | 71 | -6.364380037776194 | 0.08685748358367104
optimization step | 72 | -6.4151614705459235 | 0.03360970783670722
optimization step | 73 | -6.520725336394898 | 0.06062314105484862
optimization step | 74 | -6.71130774133398 | 0.02968714653627995
optimization step | 75 | -6.449203117435718 | 0.02839579521875277
optimization step | 76 | -6.605581356136062 | 0.048791587495080724
optimization step | 77 | -6.550121319569799 | 0.05866082781861692
optimization step | 78 | -6.625499330651917 | 0.01946749490733942
optimization step | 79 | -6.777611936899158 | 0.02965533789360993
optimization step | 80 | -6.624974137883564 | 0.052488812044150616
optimization step | 81 | -6.753231658035041 | 0.06278320271518122
optimization step | 82 | -6.691207123015843 | 0.051034008779613574
optimization step | 83 | -6.697190237197054 | 0.0538343672428332
optimization step | 84 | -6.795016383626319 | 0.037078231137357365
optimization step | 85 | -6.8619845692415895 | 0.058144590104416145
optimization step | 86 | -6.936491956782062 | 0.04103206651918714
optimization step | 87 | -6.841839837829818 | 0.06366239214778933
optimization step | 88 | -6.855555876223937 | 0.02440775789444962
optimization step | 89 | -6.8900672518511 | 0.01938819823130429
optimization step | 90 | -6.917783324838131 | 0.06502858754911242
optimization step | 91 | -6.947631434033951 | 0.025872074003336295
optimization step | 92 | -7.057329455655642 | 0.027608622638474075
optimization step | 93 | -6.932820908507631 | 0.04126286179724076
optimization step | 94 | -6.980010305619521 | 0.034529186624772515
optimization step | 95 | -7.005526691344654 | 0.029593890480471816
optimization step | 96 | -6.981124925179847 | 0.054239050188098534
optimization step | 97 | -7.065178441545735 | 0.044939282247226535
optimization step | 98 | -7.146601465214497 | 0.06198614193582842
optimization step | 99 | -7.161305550324542 | 0.034413581038572956
optimization step | 100 | -7.093298124162263 | 0.03764084568650938
optimization step | 101 | -7.107293609846828 | 0.031039782740862377
optimization step | 102 | -7.168164039548027 | 0.03504348888908298
optimization step | 103 | -7.231934804647187 | 0.04301005184222264
optimization step | 104 | -7.117703468808969 | 0.048797827996846885
optimization step | 105 | -7.209721176519217 | 0.03720462588018278
optimization step | 106 | -7.306595727623417 | 0.03411319394371843
optimization step | 107 | -7.2143295879922915 | 0.03119258654456355
optimization step | 108 | -7.334136526511671 | 0.04044215826370536
optimization step | 109 | -7.240186304203031 | 0.039664960059667013
optimization step | 110 | -7.325875668784074 | 0.014630576702965066
optimization step | 111 | -7.339730547901796 | 0.03153899797589336
optimization step | 112 | -7.3333522623768745 | 0.02021402929268427
optimization step | 113 | -7.360794793144069 | 0.018775782506117774
optimization step | 114 | -7.39717197077955 | 0.04193386426583546
optimization step | 115 | -7.391138045080448 | 0.045529137406009446
optimization step | 116 | -7.397221816858672 | 0.05836912992922437
optimization step | 117 | -7.397072536774759 | 0.04972478504573548
optimization step | 118 | -7.529814707108441 | 0.06909244475366404
optimization step | 119 | -7.4896453363302555 | 0.02103577378834082
optimization step | 120 | -7.422075928610544 | 0.024710908113011354
optimization step | 121 | -7.466190756924919 | 0.04234426507797888
optimization step | 122 | -7.533401554591632 | 0.03341462956546507
optimization step | 123 | -7.551273223042768 | 0.02909008496535029
optimization step | 124 | -7.519818889447626 | 0.039302948033523855
optimization step | 125 | -7.530293557728788 | 0.013975177413149594
optimization step | 126 | -7.546509320829145 | 0.03890681370668662
optimization step | 127 | -7.572705225817094 | 0.01203811557424265
optimization step | 128 | -7.530220830766774 | 0.027221105838463825
optimization step | 129 | -7.54402179971076 | 0.03761274394238694
optimization step | 130 | -7.543962186149915 | 0.038320166944330575
optimization step | 131 | -7.536318790555844 | 0.0546774285559157
optimization step | 132 | -7.599930666827961 | 0.038202076700965135
optimization step | 133 | -7.650279972671777 | 0.04031171800868156
optimization step | 134 | -7.6132531864662925 | 0.029653971381893074
optimization step | 135 | -7.701707039857302 | 0.015724346780682585
optimization step | 136 | -7.647742617217025 | 0.035994965757277717
optimization step | 137 | -7.724655532778525 | 0.035210125392459234
optimization step | 138 | -7.7086512778323195 | 0.021997902995837818
optimization step | 139 | -7.680262103415878 | 0.026038926535752075
optimization step | 140 | -7.726200310762155 | 0.02130241049428171
optimization step | 141 | -7.686457272121899 | 0.034503746675290464
optimization step | 142 | -7.694581472163691 | 0.030281144049443515
optimization step | 143 | -7.845740135470022 | 0.05923115759819233
optimization step | 144 | -7.667587678495723 | 0.027896724614316228
optimization step | 145 | -7.742500338662273 | 0.029915592575477025
optimization step | 146 | -7.840900901121742 | 0.05452809526938749
optimization step | 147 | -7.821557524646883 | 0.03249742109002569
optimization step | 148 | -7.781281607440791 | 0.027523335317685957
optimization step | 149 | -7.839826742553481 | 0.024264784784483826
optimization step | 150 | -7.7948957189574966 | 0.02012849927263907
optimization step | 151 | -7.799633076620718 | 0.040212396160352916
optimization step | 152 | -7.846751101579452 | 0.02416525771963804
optimization step | 153 | -7.866366949069411 | 0.02836309261576532
optimization step | 154 | -7.866789848748669 | 0.026315487848275
optimization step | 155 | -7.837043628511803 | 0.02641627394463962
optimization step | 156 | -7.879844706293314 | 0.02211136881837676
optimization step | 157 | -7.826372151088092 | 0.021296225689971107
optimization step | 158 | -7.886322726158501 | 0.01754519086876716
optimization step | 159 | -7.888145612099812 | 0.02742129838518711
optimization step | 160 | -7.858482875910316 | 0.007283598436455911
optimization step | 161 | -7.883493516820538 | 0.02564718873907164
optimization step | 162 | -7.929951835046578 | 0.0400128067158941
optimization step | 163 | -7.935727312287215 | 0.013164577029499198
optimization step | 164 | -7.908139591158921 | 0.016225424835619923
optimization step | 165 | -7.887274681041413 | 0.04463016095751775
optimization step | 166 | -7.949578003477987 | 0.009904922098489356
optimization step | 167 | -7.9317025776354315 | 0.009733242845749203
optimization step | 168 | -7.9561335482229465 | 0.02616282111726522
optimization step | 169 | -7.98581125492141 | 0.02254878052279932
optimization step | 170 | -8.005935305166542 | 0.019576569363763972
optimization step | 171 | -7.971714194560231 | 0.01661826550875264
optimization step | 172 | -7.995929176432912 | 0.02187218985946137
optimization step | 173 | -7.982652335356572 | 0.019221942067844222
optimization step | 174 | -7.97878593137028 | 0.03767374013913183
optimization step | 175 | -7.972559230428257 | 0.020986385769027806
optimization step | 176 | -7.9421025333719655 | 0.023106545595664805
optimization step | 177 | -8.006020055781248 | 0.0126888557034677
optimization step | 178 | -8.002471910403528 | 0.03088825656067885
optimization step | 179 | -8.050878820599227 | 0.02451660250023048
optimization step | 180 | -8.034635373807271 | 0.02063720181454608
optimization step | 181 | -8.013739188419862 | 0.005629994857607694
optimization step | 182 | -8.073147515925083 | 0.01920475074625459
optimization step | 183 | -8.015742966459058 | 0.027292104685373377
optimization step | 184 | -8.006511657654126 | 0.027099589946713486
optimization step | 185 | -8.006980034313377 | 0.021203494758392845
optimization step | 186 | -8.051014825628348 | 0.03565512129205927
optimization step | 187 | -8.03727265292061 | 0.019927313217494278
optimization step | 188 | -8.020135146574223 | 0.03016786977636548
optimization step | 189 | -8.09089170441948 | 0.008409912556529505
optimization step | 190 | -8.032032188337125 | 0.015599297810421025
optimization step | 191 | -8.076328261936863 | 0.024304426300677726
optimization step | 192 | -8.044667460578827 | 0.020224370303556973
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0006,
        "n_blocks": 5,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 5,
        "n_optimization_steps": 500,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+1)
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -8.115704827507974 | 0.02326925155231172
optimization step | 1 | -8.073678430779905 | 0.02851926015257702
optimization step | 2 | 2.895310555910787 | 0.24298472044146946
optimization step | 3 | -6.165873039718842 | 0.07850772388941119
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0004,
        "n_blocks": 5,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 5,
        "n_optimization_steps": 500,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+1)
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -8.11473118144312 | 0.015144802212847398
optimization step | 1 | -8.018188986106235 | 0.0438193037150075
optimization step | 2 | -1.9923147327287651 | 0.3086742536897916
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0002,
        "n_blocks": 5,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 5,
        "n_optimization_steps": 500,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+1)
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -8.11473118144312 | 0.015144802212847398
optimization step | 1 | -8.063115661286965 | 0.02286488132998444
optimization step | 2 | -8.03692117592364 | 0.03622503156026465
optimization step | 3 | -8.001993760237232 | 0.013527355657237254
optimization step | 4 | -8.015750399839355 | 0.0361873643083911
optimization step | 5 | -8.049761780746277 | 0.016779743207663494
optimization step | 6 | -8.047056796492095 | 0.01746691685807964
optimization step | 7 | -8.012911731238816 | 0.015716431827232383
optimization step | 8 | -8.074622563782702 | 0.01903152584096015
optimization step | 9 | -7.977022960837194 | 0.04988692365789496
optimization step | 10 | -7.991082713698705 | 0.017846439947742147
optimization step | 11 | -7.955616579914205 | 0.016677908869972283
optimization step | 12 | -7.986483208662321 | 0.02481790487538495
optimization step | 13 | -7.9540622103038645 | 0.014095327790562282
optimization step | 14 | -7.9567029660064295 | 0.03912053921455299
optimization step | 15 | -7.951352321946428 | 0.046411378680542396
optimization step | 16 | -7.938899728674412 | 0.03394924023391512
optimization step | 17 | -7.940034797755541 | 0.022122209928534024
optimization step | 18 | -7.9433120274550815 | 0.02435204746730279
optimization step | 19 | -7.951647895635031 | 0.03696125955678434
optimization step | 20 | -7.898186787008882 | 0.028398460070205068
optimization step | 21 | -7.921034894993686 | 0.018264468152837988
optimization step | 22 | -7.998823350318868 | 0.025723279108334152
optimization step | 23 | -8.016866901439599 | 0.030325995912900288
optimization step | 24 | -7.918375492269829 | 0.025714726193049937
optimization step | 25 | -7.946331945482337 | 0.010872394116773513
optimization step | 26 | -8.009218826700952 | 0.02768911701899786
optimization step | 27 | -7.94071680877745 | 0.021004924469937562
optimization step | 28 | -7.953207532881931 | 0.04628833427350498
optimization step | 29 | -7.990799268513013 | 0.038299223891055806
optimization step | 30 | -7.973347321037305 | 0.035735371743394896
optimization step | 31 | -7.892408503485276 | 0.01944860081123202
optimization step | 32 | -7.997866460008775 | 0.03087674532878666
optimization step | 33 | -7.978199821778534 | 0.01543943770992588
optimization step | 34 | -7.854496909212249 | 0.025936273010147493
optimization step | 35 | -7.753209914148488 | 0.04790486683368404
optimization step | 36 | -7.340437584249761 | 0.05892707968406359
optimization step | 37 | -7.328926973745068 | 0.04686635381574514
optimization step | 38 | -7.352853563795082 | 0.06367471108594076
optimization step | 39 | -7.330226458287858 | 0.07635952011695218
optimization step | 40 | -7.470521283553363 | 0.058943457361252394
optimization step | 41 | -7.434599431202574 | 0.042248976606081394
optimization step | 42 | -7.380334811745111 | 0.07222636566466632
optimization step | 43 | -6.263752763611873 | 0.09315165218619757
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0005,
        "n_blocks": 5,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 5,
        "n_optimization_steps": 500,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+1)
## Wave Function Parameters
creating wave function parameters file: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | 0.6494319591486027 | 0.32092829966538144
optimization step | 1 | -0.13812396152791664 | 0.12075425139798471
optimization step | 2 | -1.5452623077715428 | 0.2877342424296776
optimization step | 3 | -1.86411075794859 | 0.10196216453296066
optimization step | 4 | -2.2064982749122013 | 0.16749009139542392
optimization step | 5 | -2.78948981524837 | 0.09217489522967162
optimization step | 6 | -3.0017612693492577 | 0.052006129848977276
optimization step | 7 | -3.4078187664462987 | 0.14178556765403816
optimization step | 8 | -3.5531286490016982 | 0.11190577109751523
optimization step | 9 | -3.640383952416954 | 0.12084679266411809
optimization step | 10 | -3.9607861716554673 | 0.07593972980800587
optimization step | 11 | -4.106948776585558 | 0.12977743662950672
optimization step | 12 | -4.336350883609798 | 0.03218626913284745
optimization step | 13 | -4.733070093324781 | 0.07035532335511592
optimization step | 14 | -4.677619241232089 | 0.05201402922703686
optimization step | 15 | -4.854266769480991 | 0.04242475814679876
optimization step | 16 | -5.04627806885258 | 0.06975977649064057
optimization step | 17 | -5.262709204919875 | 0.03448692397279912
optimization step | 18 | -5.324887484783805 | 0.030769145180936586
optimization step | 19 | -5.52729655778229 | 0.0790064340780917
optimization step | 20 | -5.450044713691708 | 0.060130509088488986
optimization step | 21 | -5.602266603057754 | 0.05236479929798435
optimization step | 22 | -5.882949457994668 | 0.06586688110757952
optimization step | 23 | -5.963846919495339 | 0.0904907324159435
optimization step | 24 | -5.902225566798906 | 0.0577414818397779
optimization step | 25 | -6.17810988547435 | 0.045139548679249825
optimization step | 26 | -6.305085697626628 | 0.09345248105891518
optimization step | 27 | -6.262804514719346 | 0.03789751802171416
optimization step | 28 | -6.393387858383363 | 0.041762785522915485
optimization step | 29 | -6.467895519355172 | 0.05268718763159169
optimization step | 30 | -6.637733258289205 | 0.07942223442301656
optimization step | 31 | -6.526297836113128 | 0.041164422507968146
optimization step | 32 | -6.8640225657156595 | 0.06708049249510774
optimization step | 33 | -6.864291696652023 | 0.041634494514288134
optimization step | 34 | -6.774250191646351 | 0.02265580778505987
optimization step | 35 | -6.900301343415558 | 0.04667031272311731
optimization step | 36 | -6.92751041314316 | 0.011614153280215842
optimization step | 37 | -6.9950662862178605 | 0.05752086110408309
optimization step | 38 | -6.965053384356051 | 0.023052786359652713
optimization step | 39 | -7.008055559072993 | 0.049171993667641994
optimization step | 40 | -7.227385197742912 | 0.045904311451232346
optimization step | 41 | -7.198947450114129 | 0.03905191928239672
optimization step | 42 | -7.215139010399248 | 0.07749032854995772
optimization step | 43 | -7.210920148201898 | 0.05690167211230452
optimization step | 44 | -7.2839418405532115 | 0.04484304941506954
optimization step | 45 | -7.399012999080386 | 0.05818118975109443
optimization step | 46 | -7.393329427942805 | 0.026543396780882442
optimization step | 47 | -7.469667726362255 | 0.04243918143743587
optimization step | 48 | -7.483825842817424 | 0.07541300496682475
optimization step | 49 | -7.508230624224287 | 0.03589129214222901
optimization step | 50 | -7.510530888773528 | 0.06506986183788323
optimization step | 51 | -7.589358485868644 | 0.035980432807089915
optimization step | 52 | -7.597858590523005 | 0.038816026903304206
optimization step | 53 | -7.602406665567531 | 0.051011688776498886
optimization step | 54 | -7.68956894062966 | 0.049409687125552264
optimization step | 55 | -7.684882632806456 | 0.03772145240688189
optimization step | 56 | -7.659360349400552 | 0.03410690894646593
optimization step | 57 | -7.73950816646865 | 0.005526240919509835
optimization step | 58 | -7.776827044238355 | 0.022944905741453123
optimization step | 59 | -7.76302284819315 | 0.0162538539378301
optimization step | 60 | -7.8046126643563785 | 0.035270220749581624
optimization step | 61 | -7.855073532306628 | 0.028392563483672317
optimization step | 62 | -7.8591779750431545 | 0.019541219107109587
optimization step | 63 | -7.871898034596977 | 0.01626487586121649
optimization step | 64 | -7.879094080984631 | 0.023203276337671087
optimization step | 65 | -7.939626472144203 | 0.01671219590363978
optimization step | 66 | -7.932191079907833 | 0.02346158692576654
optimization step | 67 | -7.983281100098343 | 0.016224033621586625
optimization step | 68 | -7.9301701621885785 | 0.027871795649040517
optimization step | 69 | -7.9677973485933276 | 0.02424057445281782
optimization step | 70 | -7.96428303149127 | 0.01399007191083583
optimization step | 71 | -7.9631384481198015 | 0.013616784922712425
optimization step | 72 | -7.995789681208573 | 0.01762546767803221
optimization step | 73 | -8.040705693860446 | 0.01822469117605013
optimization step | 74 | -8.075245028067096 | 0.0279434141320089
optimization step | 75 | -7.965358112395106 | 0.030430416895377975
optimization step | 76 | -8.017918942760067 | 0.012162263497423977
optimization step | 77 | -8.030050546910502 | 0.019749922463963693
optimization step | 78 | -8.047787304968471 | 0.020491144799255
optimization step | 79 | -8.121479775320179 | 0.010064572974218615
optimization step | 80 | -8.06281266479858 | 0.029264707442032092
optimization step | 81 | -8.112179148896248 | 0.020613205508075085
optimization step | 82 | -8.046723957605286 | 0.021948639448489553
optimization step | 83 | -8.108552526453295 | 0.01812411495962527
optimization step | 84 | -8.115037838204344 | 0.017110827637182326
optimization step | 85 | -8.141795501498526 | 0.010061091406698024
optimization step | 86 | -8.187977718905515 | 0.020999215660133914
optimization step | 87 | -8.127416397579488 | 0.04124202179342557
optimization step | 88 | -8.161863510076525 | 0.015455394807584715
optimization step | 89 | -8.172742658580413 | 0.027266690056322363
optimization step | 90 | -8.16504650496598 | 0.013268758660358385
optimization step | 91 | -8.176183424943671 | 0.015179488662999008
optimization step | 92 | -8.204402292825673 | 0.012905525488739559
optimization step | 93 | -8.169135235546374 | 0.011681230774680964
optimization step | 94 | -8.18744495173482 | 0.01665227457856033
optimization step | 95 | -8.194204956845091 | 0.028409181097839454
optimization step | 96 | -8.223707036416727 | 0.023148460697632704
optimization step | 97 | -8.191354893227778 | 0.014178743312512974
optimization step | 98 | -8.214860150020431 | 0.03709052395962818
optimization step | 99 | -8.27061861646873 | 0.017223340418454643
optimization step | 100 | -8.271458743564468 | 0.015068982469106436
optimization step | 101 | -8.24951440415407 | 0.013475781436418277
optimization step | 102 | -8.274194235126417 | 0.022046316252093575
optimization step | 103 | -8.285967493188789 | 0.01694204167415992
optimization step | 104 | -8.233587817439378 | 0.013096562760566302
optimization step | 105 | -8.287131401811074 | 0.007524118004660385
optimization step | 106 | -8.313756124219381 | 0.005034653736415978
optimization step | 107 | -8.30362291541729 | 0.00877500756655312
optimization step | 108 | -8.309417024281265 | 0.02264796643987935
optimization step | 109 | -8.283339964108201 | 0.0074794090748455284
optimization step | 110 | -8.3262411961245 | 0.015328121177256123
optimization step | 111 | -8.34125647192589 | 0.011253957121306278
optimization step | 112 | -8.329942118170589 | 0.010555006358643763
optimization step | 113 | -8.31031589407245 | 0.021188050935937654
optimization step | 114 | -8.368860708810976 | 0.008703144987389364
optimization step | 115 | -8.324965353117559 | 0.012879270257706068
optimization step | 116 | -8.332721321189213 | 0.006008260325090927
optimization step | 117 | -8.329106549376252 | 0.012934231563372218
optimization step | 118 | -8.358155319859069 | 0.012046999085122287
optimization step | 119 | -8.37041156827472 | 0.005725180399091581
optimization step | 120 | -8.351855615892566 | 0.008861194806573363
optimization step | 121 | -8.381218936723538 | 0.01569857281251229
optimization step | 122 | -8.352885677111479 | 0.007711908340979981
optimization step | 123 | -8.377480624492323 | 0.011881055921030262
optimization step | 124 | -8.359541725150152 | 0.018150504937179384
optimization step | 125 | -8.364982780936879 | 0.00870528335505171
optimization step | 126 | -8.378169819718352 | 0.008702702658629448
optimization step | 127 | -8.386749633036988 | 0.014626000660737907
optimization step | 128 | -8.365149000881 | 0.009390120707854837
optimization step | 129 | -8.387912474690491 | 0.008000917519510034
optimization step | 130 | -8.378979577581585 | 0.022654725737755836
optimization step | 131 | -8.371535992004373 | 0.026897455480383588
optimization step | 132 | -8.368999944798485 | 0.007556250281151979
optimization step | 133 | -8.35076690995711 | 0.01722114595513705
optimization step | 134 | -8.043652052395078 | 0.024127264322216728
optimization step | 135 | -5.595108890490692 | 0.05753325774886429
optimization step | 136 | -5.065328607144399 | 0.14490954787246213
optimization step | 137 | -5.029621941977893 | 0.04943061199155084
optimization step | 138 | -0.8040106527868668 | 0.1296073029790438
optimization step | 139 | 12.492894974439457 | 0.36445125446893784
optimization step | 140 | 11.522212566059975 | 0.30809683755050915
optimization step | 141 | 7.9718245896745685 | 0.14887149473968705
optimization step | 142 | 6.801265381162537 | 0.19340306920401193
optimization step | 143 | 8.545579525620553 | 0.22400658267429638
optimization step | 144 | 1.5812111217458122 | 0.2194457282574196
optimization step | 145 | 0.8418703478059145 | 0.08953389976267064
optimization step | 146 | 0.9603786184150351 | 0.14039893844784976
optimization step | 147 | -2.332736156227301 | 0.18667315185593197
optimization step | 148 | -2.90490965719397 | 0.10500830394135832
optimization step | 149 | -3.572175159125217 | 0.1025230198960875
optimization step | 150 | -4.508817957742989 | 0.07539941826213148
optimization step | 151 | -5.106045810990956 | 0.1020207919608783
optimization step | 152 | -5.791819965477912 | 0.0818070105683704
optimization step | 153 | -6.184742989023058 | 0.05401705016414225
optimization step | 154 | -7.2703566784637985 | 0.03445353065999211
optimization step | 155 | -7.758859570577025 | 0.06741140072973444
optimization step | 156 | -7.308752794076151 | 0.043870624150072916
optimization step | 157 | -7.401064922651332 | 0.05586100818626479
optimization step | 158 | -7.29690139144585 | 0.02847935820451364
optimization step | 159 | -5.845698153775401 | 0.1317420490247804
optimization step | 160 | -1.8971361942127032 | 0.15332153964300754
optimization step | 161 | 11.628508891910222 | 0.35653311832924683
optimization step | 162 | -3.0918568292966597 | 0.17821323125368418
optimization step | 163 | -3.746987587196648 | 0.12085967780589066
optimization step | 164 | -3.8461714495221706 | 0.15694139312576147
optimization step | 165 | -3.9916435462994238 | 0.11699051101442648
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0001,
        "n_blocks": 5,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 5,
        "n_optimization_steps": 500,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+1)
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -4.53224717733306 | 0.11970931074871183
optimization step | 1 | -4.5892964464739645 | 0.04510577753068895
optimization step | 2 | -4.852776466842295 | 0.11641646316015804
optimization step | 3 | -4.611969882955893 | 0.06449948063783814
optimization step | 4 | -4.7402501587801495 | 0.05637717986389253
optimization step | 5 | -4.814200343724393 | 0.0945093828951792
optimization step | 6 | -4.933856284446641 | 0.02893388201497483
optimization step | 7 | -5.065924033106368 | 0.09343404410379845
optimization step | 8 | -5.029202987538298 | 0.06545253319049751
optimization step | 9 | -4.858093929417032 | 0.1290153587909429
optimization step | 10 | -5.0942749349696035 | 0.03841113087552523
optimization step | 11 | -4.977339755918289 | 0.11360832984416151
optimization step | 12 | -5.052327854207042 | 0.08722990706665816
optimization step | 13 | -5.176976048963465 | 0.10089551931756346
optimization step | 14 | -5.050121104416922 | 0.13930013644602116
optimization step | 15 | -5.16627329364529 | 0.03409179120220985
optimization step | 16 | -5.239964872900567 | 0.08247929207132514
optimization step | 17 | -5.325520614960468 | 0.0724206099060609
optimization step | 18 | -5.199725951043667 | 0.03904997774249091
optimization step | 19 | -5.4382093768210895 | 0.09201971728387076
optimization step | 20 | -5.08708198623367 | 0.0665795505345856
optimization step | 21 | -5.272477285015856 | 0.08050317085409588
optimization step | 22 | -5.399920056242795 | 0.025227993615709536
optimization step | 23 | -5.367381348709871 | 0.10965673772100692
optimization step | 24 | -5.218365747190544 | 0.08214213837028655
optimization step | 25 | -5.450585020294482 | 0.08244485519337741
optimization step | 26 | -5.531541850090261 | 0.03622485256188006
optimization step | 27 | -5.458265457074268 | 0.037952624520491224
optimization step | 28 | -5.53149690532166 | 0.04828885313309414
optimization step | 29 | -5.572716893874951 | 0.10332256606135945
optimization step | 30 | -5.653785921942932 | 0.09241737328796035
optimization step | 31 | -5.462904222329409 | 0.025081468092220404
optimization step | 32 | -5.692978386541232 | 0.05543323148001663
optimization step | 33 | -5.663165118377509 | 0.07442615670009116
optimization step | 34 | -5.578719146965577 | 0.06021445268145873
optimization step | 35 | -5.676437270688683 | 0.07918954065237976
optimization step | 36 | -5.635851457124253 | 0.06757858472257669
optimization step | 37 | -5.72894783835128 | 0.044570336986159316
optimization step | 38 | -5.599303699206304 | 0.04608563931552864
optimization step | 39 | -5.663208041529652 | 0.07641192215855482
optimization step | 40 | -5.874650901478054 | 0.08750535757104239
optimization step | 41 | -5.849434666083891 | 0.06209799654442859
optimization step | 42 | -5.879975327149518 | 0.09160443012373398
optimization step | 43 | -5.762942642722272 | 0.09074837807297302
optimization step | 44 | -5.804237278875291 | 0.09155500926340376
optimization step | 45 | -5.97328583132096 | 0.08262270848790708
optimization step | 46 | -6.030634008878863 | 0.048562673622125054
optimization step | 47 | -6.129136817293573 | 0.09780917813366988
optimization step | 48 | -6.071993937486501 | 0.104875485157401
optimization step | 49 | -6.03046934425069 | 0.07907908570693414
optimization step | 50 | -6.070240894261229 | 0.12987932834873273
optimization step | 51 | -6.201810662862322 | 0.08254861537013443
optimization step | 52 | -6.024469858327924 | 0.05112711621749647
optimization step | 53 | -6.013320284860703 | 0.11601944323033034
optimization step | 54 | -6.103610166240759 | 0.06119491161524957
optimization step | 55 | -6.050837485803739 | 0.0515374863438559
optimization step | 56 | -5.954681411605471 | 0.027322378353201456
optimization step | 57 | -6.168564633027648 | 0.04166983724183261
optimization step | 58 | -6.106400412455733 | 0.067591522343
optimization step | 59 | -6.255806937110556 | 0.0418675244778672
optimization step | 60 | -6.298596379995894 | 0.08264246269812311
optimization step | 61 | -6.1521981114017885 | 0.08628259398264167
optimization step | 62 | -6.239744247341456 | 0.0370388746162175
optimization step | 63 | -6.322750300733035 | 0.05566778031175954
optimization step | 64 | -6.3605708916440715 | 0.0534167407850551
optimization step | 65 | -6.373895333871222 | 0.07027222664082217
optimization step | 66 | -6.399913102895436 | 0.05288901021746917
optimization step | 67 | -6.3772407042813315 | 0.08548806642887685
optimization step | 68 | -6.356837797740313 | 0.10440086399561448
optimization step | 69 | -6.428699200344822 | 0.08075472321565397
optimization step | 70 | -6.369241158261497 | 0.026889695454856102
optimization step | 71 | -6.297491419574323 | 0.0687347135197492
optimization step | 72 | -6.383160971979029 | 0.03295687146291431
optimization step | 73 | -6.3746968171360034 | 0.08829172424599357
optimization step | 74 | -6.611895645008856 | 0.03391345653127462
optimization step | 75 | -6.350557382394081 | 0.041333917559090314
optimization step | 76 | -6.446420849136517 | 0.06021975159197615
optimization step | 77 | -6.398308624372689 | 0.04530142054245769
optimization step | 78 | -6.512298427644637 | 0.036061636970555075
optimization step | 79 | -6.599096887554128 | 0.01634528720987317
optimization step | 80 | -6.4610933085646876 | 0.06707006505351303
optimization step | 81 | -6.5743938841750165 | 0.07199271815348822
optimization step | 82 | -6.535004502524027 | 0.08381530909490509
optimization step | 83 | -6.5223885296649495 | 0.05705559062752498
optimization step | 84 | -6.58871297129001 | 0.04221053769897706
optimization step | 85 | -6.624815101087291 | 0.04259550965126546
optimization step | 86 | -6.742886767560988 | 0.06553914306121753
optimization step | 87 | -6.5579451008975855 | 0.06609390727512274
optimization step | 88 | -6.629788884544001 | 0.0322420316367507
optimization step | 89 | -6.6173286606858905 | 0.044395132779128633
optimization step | 90 | -6.661921566559232 | 0.07384150677496785
optimization step | 91 | -6.6742150845936 | 0.028494837117354746
optimization step | 92 | -6.741743488354904 | 0.02704775011166975
optimization step | 93 | -6.643044489485336 | 0.059476948151240314
optimization step | 94 | -6.724691293168901 | 0.028205563052097295
optimization step | 95 | -6.658898046724026 | 0.028267268782817302
optimization step | 96 | -6.645831708645377 | 0.07007969644241285
optimization step | 97 | -6.745872165934739 | 0.04252778038584592
optimization step | 98 | -6.815893108347313 | 0.062424347215294385
optimization step | 99 | -6.779811224140838 | 0.06852882382598247
optimization step | 100 | -6.761835851121108 | 0.049673354357346926
optimization step | 101 | -6.70384451292652 | 0.03358250840859434
optimization step | 102 | -6.809647541598489 | 0.03912660817342981
optimization step | 103 | -6.811853480831468 | 0.05705059264072837
optimization step | 104 | -6.775560130665899 | 0.03620456457895729
optimization step | 105 | -6.765240141705576 | 0.05054858519232686
optimization step | 106 | -6.86363685663208 | 0.035934810581946024
optimization step | 107 | -6.859944131121084 | 0.061590108665065785
optimization step | 108 | -6.917723181544306 | 0.03653958879085123
optimization step | 109 | -6.814838649457103 | 0.052013545278859784
optimization step | 110 | -6.884619499936571 | 0.03548787085737171
optimization step | 111 | -6.902936580586106 | 0.04028498882624751
optimization step | 112 | -6.8657417463680535 | 0.030087157861325136
optimization step | 113 | -6.887238873714897 | 0.017864520660504063
optimization step | 114 | -6.996117762304749 | 0.051285141679117116
optimization step | 115 | -6.950851806790384 | 0.02182850834771033
optimization step | 116 | -6.909436951226733 | 0.06999872483163921
optimization step | 117 | -6.90469597087297 | 0.07842579753671826
optimization step | 118 | -7.026164737992412 | 0.07049005973117274
optimization step | 119 | -6.995973346672732 | 0.04998594134697048
optimization step | 120 | -6.9232370600411155 | 0.05837493175381189
optimization step | 121 | -6.998964446486186 | 0.055286659959649126
optimization step | 122 | -7.008014220035795 | 0.04049938201705036
optimization step | 123 | -7.068687678701376 | 0.08429675195624417
optimization step | 124 | -7.017547278791018 | 0.04949355931781746
optimization step | 125 | -7.017801727350204 | 0.04613005161722055
optimization step | 126 | -7.01911535098466 | 0.05635595193298358
optimization step | 127 | -6.979749068271381 | 0.02490890486848246
optimization step | 128 | -6.979676451324724 | 0.061620829745366146
optimization step | 129 | -7.048076975504128 | 0.03174846388847903
optimization step | 130 | -6.984196192459383 | 0.03646023687039222
optimization step | 131 | -7.007749802013642 | 0.05562494361342228
optimization step | 132 | -7.035150385055632 | 0.055749522644049566
optimization step | 133 | -7.069493471434089 | 0.03619373202056114
optimization step | 134 | -7.029845181770874 | 0.02759240059230201
optimization step | 135 | -7.162437937196141 | 0.044254379873286336
optimization step | 136 | -7.058982460648005 | 0.039694405910577095
optimization step | 137 | -7.160952409679193 | 0.027495381434528652
optimization step | 138 | -7.1168915413508556 | 0.01236641104022651
optimization step | 139 | -7.069935624106824 | 0.021297158907025136
optimization step | 140 | -7.153656763354678 | 0.039611680680824116
optimization step | 141 | -7.09391565366438 | 0.050521803626524774
optimization step | 142 | -7.087631505121554 | 0.022346700507883267
optimization step | 143 | -7.2759852923807955 | 0.05840021365286491
optimization step | 144 | -7.083253866226022 | 0.040250608877215245
optimization step | 145 | -7.186178788807739 | 0.01857536871199575
optimization step | 146 | -7.342803203272487 | 0.06545365155705073
optimization step | 147 | -7.228123581765085 | 0.05074855256987383
optimization step | 148 | -7.224185921087697 | 0.0345622415154289
optimization step | 149 | -7.256643120697875 | 0.04921809252022377
optimization step | 150 | -7.183341634244994 | 0.026181402191602945
optimization step | 151 | -7.205514115179649 | 0.07911129699810006
optimization step | 152 | -7.276425638563768 | 0.02802055132154629
optimization step | 153 | -7.276134141290396 | 0.015863489006051366
optimization step | 154 | -7.264267652116695 | 0.04134342354583448
optimization step | 155 | -7.229440653517993 | 0.019545238903387118
optimization step | 156 | -7.2992943588472485 | 0.03773207812052628
optimization step | 157 | -7.226887619557876 | 0.025636501579595943
optimization step | 158 | -7.296635303940898 | 0.03563793634612914
optimization step | 159 | -7.253223368830741 | 0.0342449267396395
optimization step | 160 | -7.239452483324861 | 0.011966871587999296
optimization step | 161 | -7.2744069414912165 | 0.06658146808623318
optimization step | 162 | -7.306264731708223 | 0.05873749836679882
optimization step | 163 | -7.345627639822202 | 0.026261662524544448
optimization step | 164 | -7.249353391238799 | 0.024894253924134115
optimization step | 165 | -7.251833476146817 | 0.04648515552397624
optimization step | 166 | -7.277729301141553 | 0.023825300611111624
optimization step | 167 | -7.344792304066566 | 0.039562949374833754
optimization step | 168 | -7.3887064676726695 | 0.028218110626364205
optimization step | 169 | -7.414918131577194 | 0.02294450841875811
optimization step | 170 | -7.404977187994045 | 0.029161221981187903
optimization step | 171 | -7.423287830854408 | 0.03245478568723265
optimization step | 172 | -7.416222727137816 | 0.02749390784280227
optimization step | 173 | -7.378058946313621 | 0.04922843751367847
optimization step | 174 | -7.33826020210477 | 0.04546058053331402
optimization step | 175 | -7.3524852388873425 | 0.043185044450905125
optimization step | 176 | -7.282973533817319 | 0.008750680319505719
optimization step | 177 | -7.389189993829737 | 0.014763420595067227
optimization step | 178 | -7.405895502534335 | 0.04939001308803871
optimization step | 179 | -7.455257052813801 | 0.04611401444625739
optimization step | 180 | -7.456347304951322 | 0.04222178087451921
optimization step | 181 | -7.432177850982728 | 0.01826914162571934
optimization step | 182 | -7.479610817812301 | 0.037746630102526685
optimization step | 183 | -7.386655233877853 | 0.04541090796209791
optimization step | 184 | -7.447406178401343 | 0.02582518390741892
optimization step | 185 | -7.397350051595988 | 0.03671308466983245
optimization step | 186 | -7.444759574648039 | 0.05401181274736468
optimization step | 187 | -7.415924407484951 | 0.037869029818302806
optimization step | 188 | -7.424878594840713 | 0.039353707265510456
optimization step | 189 | -7.503543089987292 | 0.01812626343023406
optimization step | 190 | -7.444546108504857 | 0.02038381430915635
optimization step | 191 | -7.4470059375799735 | 0.0262034619594684
optimization step | 192 | -7.470788123957104 | 0.038399189725266854
optimization step | 193 | -7.544791477657266 | 0.04370088451330206
optimization step | 194 | -7.419086524275876 | 0.008842708264614786
optimization step | 195 | -7.4739883168397085 | 0.02164104827872338
optimization step | 196 | -7.490341233642548 | 0.024520876494468047
optimization step | 197 | -7.532451203471121 | 0.04055924553031507
optimization step | 198 | -7.547926610851965 | 0.02910045665361732
optimization step | 199 | -7.528264974972492 | 0.03151221116700758
optimization step | 200 | -7.462116319388341 | 0.010108747113859692
optimization step | 201 | -7.484578351098904 | 0.05647010442555676
optimization step | 202 | -7.553369141611418 | 0.030122114156140464
optimization step | 203 | -7.539924135756969 | 0.033455994410139564
optimization step | 204 | -7.504176197042604 | 0.03101517864255288
optimization step | 205 | -7.564330917144664 | 0.01427201041298237
optimization step | 206 | -7.507160910032743 | 0.0335528726609933
optimization step | 207 | -7.614709686327347 | 0.02669615511655698
optimization step | 208 | -7.5155516094486305 | 0.02633064452606022
optimization step | 209 | -7.541380041044088 | 0.03616437648383207
optimization step | 210 | -7.51899691952994 | 0.01637786618679648
optimization step | 211 | -7.520858841621549 | 0.03897303835259638
optimization step | 212 | -7.581395929368918 | 0.03819301840313532
optimization step | 213 | -7.560184327539074 | 0.03391365983178061
optimization step | 214 | -7.601044343572264 | 0.02379333552039623
optimization step | 215 | -7.6209890985465165 | 0.03569751990381469
optimization step | 216 | -7.635824441118852 | 0.05334397854440446
optimization step | 217 | -7.5563725467425 | 0.028903106920470104
optimization step | 218 | -7.606330292609317 | 0.034760326049993705
optimization step | 219 | -7.588010099377617 | 0.016916888736638033
optimization step | 220 | -7.617008917217367 | 0.04536092394540238
optimization step | 221 | -7.559971016332064 | 0.04057351177356761
optimization step | 222 | -7.571229061883633 | 0.03282940951676117
optimization step | 223 | -7.628438041357003 | 0.03902508019931596
optimization step | 224 | -7.5234986152428345 | 0.0280432803417397
optimization step | 225 | -7.609893327930266 | 0.01595679203472482
optimization step | 226 | -7.719620544635598 | 0.03898146159842496
optimization step | 227 | -7.649019293460145 | 0.03409332153033118
optimization step | 228 | -7.607153163839429 | 0.03833938859309195
optimization step | 229 | -7.5608989612481725 | 0.03538564665377627
optimization step | 230 | -7.540802869797112 | 0.03377635001441408
optimization step | 231 | -7.616729905593599 | 0.01480692871727488
optimization step | 232 | -7.709988950666767 | 0.024999439610805557
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0005,
        "n_blocks": 5,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 5,
        "n_optimization_steps": 500,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+1)
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -7.688599501157981 | 0.018172788680555007
optimization step | 1 | -7.648206585759371 | 0.007341702984590924
optimization step | 2 | -7.720915027059602 | 0.011534405551432824
optimization step | 3 | -7.629875815045653 | 0.025609664266376658
optimization step | 4 | -7.6626485587755635 | 0.010006808497430292
optimization step | 5 | -7.7780171285816335 | 0.017893786874155562
optimization step | 6 | -7.847915142060815 | 0.016731839619820745
optimization step | 7 | -7.7530281442746 | 0.025582498128762596
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0002,
        "n_blocks": 5,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 5,
        "n_optimization_steps": 500,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+1)
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -7.8323428144347265 | 0.014260710294939947
optimization step | 1 | -7.798310277600656 | 0.00964257440334327
optimization step | 2 | -7.804111537730057 | 0.004746114312638433
optimization step | 3 | -7.772199014608963 | 0.02425353975543344
optimization step | 4 | -7.777292211746176 | 0.015759980697537188
optimization step | 5 | -7.862692563589195 | 0.01199074617290778
optimization step | 6 | -7.89105159506862 | 0.019779568442750972
optimization step | 7 | -7.806134572099845 | 0.01918921715258724
optimization step | 8 | -7.88081021005524 | 0.013159693948178917
optimization step | 9 | -7.828093569994655 | 0.03260732289937711
optimization step | 10 | -7.83239011789429 | 0.01637768891747483
optimization step | 11 | -7.837644767508583 | 0.019558783301209928
optimization step | 12 | -7.854149661408525 | 0.015007410190690558
optimization step | 13 | -7.885121842696125 | 0.016486895824667207
optimization step | 14 | -7.847016391663966 | 0.023342287857724905
optimization step | 15 | -7.866721727298407 | 0.03970513379708095
optimization step | 16 | -7.839863247543988 | 0.026252739768392538
optimization step | 17 | -7.851197826048653 | 0.026057386208174504
optimization step | 18 | -7.916728428689295 | 0.0204288827965797
optimization step | 19 | -7.883814158023746 | 0.027116473444766177
optimization step | 20 | -7.831945576629598 | 0.02189912754120487
optimization step | 21 | -7.801817271966196 | 0.014418116686372226
optimization step | 22 | -7.927216625622364 | 0.027331260732119268
optimization step | 23 | -7.917180315035422 | 0.012503583173030436
optimization step | 24 | -7.867957957940438 | 0.026354615477218356
optimization step | 25 | -7.909428886098702 | 0.024259185664091458
optimization step | 26 | -7.975730889370546 | 0.018054048299432742
optimization step | 27 | -7.908701763268555 | 0.008589293652291899
optimization step | 28 | -7.930997973185994 | 0.029282057980579444
optimization step | 29 | -7.94799173516513 | 0.009161990376850433
optimization step | 30 | -7.96646197789272 | 0.027901865324494998
optimization step | 31 | -7.915836105229644 | 0.029241579836783438
optimization step | 32 | -8.06024297372035 | 0.033606192534861655
optimization step | 33 | -8.018086146235113 | 0.030228719082991833
optimization step | 34 | -7.9725009922828765 | 0.008758850815254808
optimization step | 35 | -7.981904084364594 | 0.012001844819419178
optimization step | 36 | -7.962660927163794 | 0.020394886614450845
optimization step | 37 | -7.929407416765723 | 0.02740549738682715
optimization step | 38 | -7.95632848909023 | 0.03764970642834997
optimization step | 39 | -7.958789472607174 | 0.015036474065031792
optimization step | 40 | -8.046257907583723 | 0.03335454864827057
optimization step | 41 | -7.948357188994083 | 0.02976662498121943
optimization step | 42 | -7.999594433694559 | 0.03539302420690571
optimization step | 43 | -7.983113151307705 | 0.019098025578621114
optimization step | 44 | -8.039365206175521 | 0.02529545539168359
optimization step | 45 | -8.020035467673052 | 0.026407961957330205
optimization step | 46 | -8.025985328676311 | 0.013784213784885329
optimization step | 47 | -8.10166637747415 | 0.025789615514814682
optimization step | 48 | -8.058544601437328 | 0.02864270117544656
optimization step | 49 | -8.055292047024622 | 0.016570348608086802
optimization step | 50 | -8.041509937927936 | 0.034983656565336474
optimization step | 51 | -8.066082755781993 | 0.03498016853915022
optimization step | 52 | -8.09458651926594 | 0.02148072509584884
optimization step | 53 | -8.059081187485784 | 0.01444337229701197
optimization step | 54 | -8.089791798560999 | 0.01997090883772875
optimization step | 55 | -8.081852834553022 | 0.030171310005770457
optimization step | 56 | -8.052863843053848 | 0.018373037755279934
optimization step | 57 | -8.110651174912721 | 0.011884745549802584
optimization step | 58 | -8.141627484875567 | 0.018594424604327853
optimization step | 59 | -8.078468068296273 | 0.0197805752235472
optimization step | 60 | -8.091694056846077 | 0.023550696777691036
optimization step | 61 | -8.156763572387364 | 0.014357626821949873
optimization step | 62 | -8.12183238587069 | 0.018366993578819595
optimization step | 63 | -8.124306885671546 | 0.01504126507379977
optimization step | 64 | -8.109784331369115 | 0.022233555003588385
optimization step | 65 | -8.152657066600991 | 0.02069698949250497
optimization step | 66 | -8.149214146238666 | 0.01413210241449825
optimization step | 67 | -8.206136588729857 | 0.030650963080161235
optimization step | 68 | -8.164620602374663 | 0.01298060270768834
optimization step | 69 | -8.154414733437868 | 0.015973597837793087
optimization step | 70 | -8.144299586478983 | 0.012607254074069423
optimization step | 71 | -8.151557858634375 | 0.009087823781504495
optimization step | 72 | -8.16865146662219 | 0.02066150112321107
optimization step | 73 | -8.189391700526922 | 0.02045769167341511
optimization step | 74 | -8.19023475330222 | 0.029678262795492447
optimization step | 75 | -8.119688243699969 | 0.03070365487038285
optimization step | 76 | -8.16149998035946 | 0.011467219793588217
optimization step | 77 | -8.147096336029133 | 0.03074779881431585
optimization step | 78 | -8.170803722501825 | 0.024873948223869208
optimization step | 79 | -8.215961237094987 | 0.010458072884415958
optimization step | 80 | -8.175841234352555 | 0.018580962663426216
optimization step | 81 | -8.19672251688809 | 0.017052562800112643
optimization step | 82 | -8.167733351877725 | 0.015376795110270158
optimization step | 83 | -8.216192849954448 | 0.008562935122789337
optimization step | 84 | -8.207006818947104 | 0.01293212157699621
optimization step | 85 | -8.22120555417759 | 0.0068188893213784576
optimization step | 86 | -8.247118389660159 | 0.018179413766931996
optimization step | 87 | -8.230478314606776 | 0.03340108512381385
optimization step | 88 | -8.232895207419487 | 0.009686620815685089
optimization step | 89 | -8.231525480322462 | 0.026202947070517603
optimization step | 90 | -8.225597273183386 | 0.02066668321378992
optimization step | 91 | -8.22672824138132 | 0.023052587224110015
optimization step | 92 | -8.251307786117797 | 0.013813091564168224
optimization step | 93 | -8.20097029010024 | 0.020408346143168127
optimization step | 94 | -8.248989608455053 | 0.01728216723920949
optimization step | 95 | -8.249832307035444 | 0.038807106557238076
optimization step | 96 | -8.2834313005544 | 0.026857348584674347
optimization step | 97 | -8.2280838509696 | 0.021093082136432392
optimization step | 98 | -8.24025953553714 | 0.02266811852503451
optimization step | 99 | -8.291250150701512 | 0.019049855087712224
optimization step | 100 | -8.263155319442735 | 0.01747194391948789
optimization step | 101 | -8.27528715092851 | 0.023351776013749197
optimization step | 102 | -8.257784658745747 | 0.02337980399568234
optimization step | 103 | -8.243017342538213 | 0.019627397756547676
optimization step | 104 | -8.216932117045634 | 0.027623658892565853
optimization step | 105 | -8.30092883450486 | 0.036375340507136804
optimization step | 106 | -8.264933429401717 | 0.008225674984166898
optimization step | 107 | -8.263675693680694 | 0.028528337392007056
optimization step | 108 | -8.244176945538523 | 0.028761478807085246
optimization step | 109 | -8.236872556488368 | 0.03012473516043093
optimization step | 110 | -8.280448118637988 | 0.02081381144054579
optimization step | 111 | -8.26981152288386 | 0.017436650926266593
optimization step | 112 | -8.291495531365598 | 0.024639335380152127
optimization step | 113 | -8.222684740940313 | 0.03280759789934753
optimization step | 114 | -8.292878301048287 | 0.02236696621993472
optimization step | 115 | -8.265495956461548 | 0.04277212570141977
optimization step | 116 | -8.294329194859973 | 0.031849614499918105
optimization step | 117 | -8.290504131840347 | 0.027073478201153314
optimization step | 118 | -8.270502896762826 | 0.021522395482517064
optimization step | 119 | -8.290169726963686 | 0.013960787243841663
optimization step | 120 | -8.27951910236466 | 0.027364368126414972
optimization step | 121 | -8.30507409158048 | 0.02924332573957407
optimization step | 122 | -8.263913703392818 | 0.025523185292158548
optimization step | 123 | -8.298079215421161 | 0.027053094927601492
optimization step | 124 | -8.272133080182813 | 0.026246249784817786
optimization step | 125 | -8.307594931495752 | 0.02540407276989936
optimization step | 126 | -8.300050942741489 | 0.013674359499290606
optimization step | 127 | -8.348234584717174 | 0.01892133110137793
optimization step | 128 | -8.292630521144778 | 0.020070495000395906
optimization step | 129 | -8.291250612672396 | 0.018625176864044007
optimization step | 130 | -8.337982545697951 | 0.028270805464672248
optimization step | 131 | -8.319412410148162 | 0.024870023835949523
optimization step | 132 | -8.309364819389392 | 0.02116056392088695
optimization step | 133 | -8.322432524961922 | 0.009446090606841862
optimization step | 134 | -8.363925621204809 | 0.008285745881855105
optimization step | 135 | -8.323709499650594 | 0.02361710621215842
optimization step | 136 | -8.320102823535377 | 0.02302567927825516
optimization step | 137 | -8.264911971204604 | 0.025868396344968108
optimization step | 138 | -8.284101465389837 | 0.025661532337323767
optimization step | 139 | -8.319757809275314 | 0.018604682081541906
optimization step | 140 | -8.328587388910979 | 0.016957221148974095
optimization step | 141 | -8.310345368367589 | 0.01371417136731613
optimization step | 142 | -8.321717291225186 | 0.013398112222523928
optimization step | 143 | -8.33987247997789 | 0.012224319502567067
optimization step | 144 | -8.364628477145981 | 0.01599110180689832
optimization step | 145 | -8.33012193997554 | 0.012820807985346292
optimization step | 146 | -8.316744941098388 | 0.025507452052828214
optimization step | 147 | -8.316741596842537 | 0.01815212209945587
optimization step | 148 | -8.347214851210667 | 0.012759677775521977
optimization step | 149 | -8.34329725390153 | 0.020843213905731572
optimization step | 150 | -8.337827447994458 | 0.012187780869497445
optimization step | 151 | -8.324337567389609 | 0.02061396049011737
optimization step | 152 | -8.326262781411762 | 0.020745750910781974
optimization step | 153 | -8.311972278783491 | 0.012248372493036083
optimization step | 154 | -8.328373248968598 | 0.014412894640067462
optimization step | 155 | -8.306084027536254 | 0.01243224221608635
optimization step | 156 | -8.332019846883748 | 0.019301117996706776
optimization step | 157 | -8.33389681435494 | 0.01887760624979592
optimization step | 158 | -8.31494970436642 | 0.01090719835984204
optimization step | 159 | -8.339374182844328 | 0.02489571039097719
optimization step | 160 | -8.336371244218016 | 0.0052032938649641735
optimization step | 161 | -8.353153949649467 | 0.020006195129062678
optimization step | 162 | -8.348574275560782 | 0.019213038310043833
optimization step | 163 | -8.319550104244348 | 0.010018794775555836
optimization step | 164 | -8.360038559726052 | 0.016753237508874773
optimization step | 165 | -8.353972545228006 | 0.01119042165667487
optimization step | 166 | -8.345745647585941 | 0.03422872648789206
optimization step | 167 | -8.363491442854187 | 0.009062272448227578
optimization step | 168 | -8.359852830490928 | 0.013342406025159374
optimization step | 169 | -8.314925076845702 | 0.004101993111006934
optimization step | 170 | -8.336795776753405 | 0.011531401476832171
optimization step | 171 | -8.283252325097525 | 0.019581216384382163
optimization step | 172 | -8.323115623605725 | 0.025129352105316045
optimization step | 173 | -8.298537728425805 | 0.04660674897622682
optimization step | 174 | -8.349815880418817 | 0.030467524430251195
optimization step | 175 | -8.312542128563232 | 0.032778576481714436
optimization step | 176 | -8.320563097671995 | 0.0329951769286656
optimization step | 177 | -8.377908924837108 | 0.03314574005985409
optimization step | 178 | -8.300686725814803 | 0.025614209517260865
optimization step | 179 | -8.293597273192693 | 0.019263476158697234
optimization step | 180 | -8.280912175967442 | 0.019212860348587418
optimization step | 181 | -8.314875538977828 | 0.02804814301933845
optimization step | 182 | -8.245593345507606 | 0.024539278478042836
optimization step | 183 | -8.2750578410509 | 0.037028777879791246
optimization step | 184 | -8.290601982518051 | 0.007447791682765619
optimization step | 185 | -8.267750851062 | 0.03301784664192547
optimization step | 186 | -8.313125373712012 | 0.025474509202045876
optimization step | 187 | -8.31422367181311 | 0.019975136034510264
optimization step | 188 | -8.25219438319414 | 0.016623303407289583
optimization step | 189 | -8.305384864161399 | 0.03515384238009305
optimization step | 190 | -8.314328044877573 | 0.012422473621789986
optimization step | 191 | -8.317174588427367 | 0.01288977723623965
optimization step | 192 | -8.290143853494666 | 0.009194890577328703
optimization step | 193 | -8.331629237682662 | 0.026575307972043185
optimization step | 194 | -8.30699325318123 | 0.01933863329971688
optimization step | 195 | -8.367153271390293 | 0.032031570449262206
optimization step | 196 | -8.307049555257064 | 0.027753129678049892
optimization step | 197 | -8.314763176974214 | 0.010449838699133724
optimization step | 198 | -8.353317751338341 | 0.011139536663193884
optimization step | 199 | -8.2921372630316 | 0.024167931796808612
optimization step | 200 | -8.349404517292758 | 0.02094672184887865
optimization step | 201 | -8.350303892010547 | 0.02920881369873837
optimization step | 202 | -8.335700982914853 | 0.015119372456042467
optimization step | 203 | -8.344647553751944 | 0.029238670314286924
optimization step | 204 | -8.325998088115679 | 0.023901118965069783
optimization step | 205 | -8.33424356316308 | 0.017892091010944283
optimization step | 206 | -8.347935479155476 | 0.023172387262799098
optimization step | 207 | -8.347329048860967 | 0.004351994816964592
optimization step | 208 | -8.381570148438271 | 0.024456495360961613
optimization step | 209 | -8.37794375142786 | 0.016239552084201136
optimization step | 210 | -8.369294271329569 | 0.022028653630301834
optimization step | 211 | -8.329962091786445 | 0.01788040270715308
optimization step | 212 | -8.359494081825684 | 0.013495961061613673
optimization step | 213 | -8.371843068590646 | 0.01974413244625525
optimization step | 214 | -8.352083600638753 | 0.009511521096177597
optimization step | 215 | -8.378725044764202 | 0.005524167747403506
optimization step | 216 | -8.3559766744758 | 0.02300205978643452
optimization step | 217 | -8.384969516758016 | 0.020129188286513926
optimization step | 218 | -8.365367549105114 | 0.01103158565223029
optimization step | 219 | -8.356768483693916 | 0.01047267553606774
optimization step | 220 | -8.354262071006072 | 0.01028937278824246
optimization step | 221 | -8.378501915333104 | 0.010683950789286738
optimization step | 222 | -8.402110850666697 | 0.011992831733728865
optimization step | 223 | -8.367265593792215 | 0.013701342890203888
optimization step | 224 | -8.374077789714878 | 0.006345473783340682
optimization step | 225 | -8.373085540786242 | 0.013831571255793575
optimization step | 226 | -8.345919037446905 | 0.013239219283048075
optimization step | 227 | -8.366254550790874 | 0.014539037285452154
optimization step | 228 | -8.379087694922385 | 0.009615622499202817
optimization step | 229 | -8.380846818290205 | 0.026090664738712268
optimization step | 230 | -8.353046619207774 | 0.016709981451223134
optimization step | 231 | -8.386513146674455 | 0.012409199169300536
optimization step | 232 | -8.357493810992812 | 0.015065241988604467
optimization step | 233 | -8.388554220517326 | 0.016749268914713202
optimization step | 234 | -8.381042225948343 | 0.013665954773967108
optimization step | 235 | -8.424381312503225 | 0.014302545300298206
optimization step | 236 | -8.374030189911839 | 0.027563141733123332
optimization step | 237 | -8.384235282890156 | 0.01916227243298412
optimization step | 238 | -8.38979622186029 | 0.010639707787414805
optimization step | 239 | -8.369320747642137 | 0.016257340955283205
optimization step | 240 | -8.368588698722837 | 0.005760773809850126
optimization step | 241 | -8.367954918436297 | 0.019163979703667993
optimization step | 242 | -8.376146933144089 | 0.016247995635566663
optimization step | 243 | -8.371601114325486 | 0.014111156328180292
optimization step | 244 | -8.372790169111948 | 0.010770679348692589
optimization step | 245 | -8.391363463820612 | 0.01623712860246411
optimization step | 246 | -8.37712254317089 | 0.029793064428480964
optimization step | 247 | -8.38518225137267 | 0.00804532277727191
optimization step | 248 | -8.393514949121656 | 0.02145011510180553
optimization step | 249 | -8.373582483034577 | 0.012582250081330026
optimization step | 250 | -8.380345489827178 | 0.023653570480377972
optimization step | 251 | -8.390233953218686 | 0.010936202635836955
optimization step | 252 | -8.391835512954538 | 0.00829371777571528
optimization step | 253 | -8.359285754176362 | 0.00542966917265222
optimization step | 254 | -8.391667032313482 | 0.012107367364630504
optimization step | 255 | -8.400164633358845 | 0.00893003417379754
optimization step | 256 | -8.366935778005418 | 0.012477894202039227
optimization step | 257 | -8.406823156487619 | 0.020870771321297627
optimization step | 258 | -8.382805907674928 | 0.012406530767253392
optimization step | 259 | -8.374444861089358 | 0.01683554686882149
optimization step | 260 | -8.38488981062421 | 0.005085497721059869
optimization step | 261 | -8.394736464738136 | 0.01379754351579612
optimization step | 262 | -8.382677776416935 | 0.0034687865354339083
optimization step | 263 | -8.389496414613848 | 0.019290688117210766
optimization step | 264 | -8.396846257189576 | 0.013570540922445248
optimization step | 265 | -8.366662489522659 | 0.010819611372480185
optimization step | 266 | -8.409424260381765 | 0.010569247909260822
optimization step | 267 | -8.381591495256878 | 0.009986218476449038
optimization step | 268 | -8.396285137747485 | 0.011753702868529364
optimization step | 269 | -8.389542402552113 | 0.006895241021924324
optimization step | 270 | -8.37182877077257 | 0.009734140021914764
optimization step | 271 | -8.376252679693872 | 0.01582880905289987
optimization step | 272 | -8.402698142171522 | 0.015076448035514601
optimization step | 273 | -8.398562201219892 | 0.018910099068347633
optimization step | 274 | -8.415925406882272 | 0.016385984575278106
optimization step | 275 | -8.390753294495372 | 0.00760026525219666
optimization step | 276 | -8.401753011776979 | 0.009003199727181468
optimization step | 277 | -8.392761736602733 | 0.012475808549466914
optimization step | 278 | -8.40703802556936 | 0.0068317999044989145
optimization step | 279 | -8.397243703508309 | 0.013094470135480688
optimization step | 280 | -8.377230867876586 | 0.012111378542552174
optimization step | 281 | -8.38184904704889 | 0.014650069900755392
optimization step | 282 | -8.385175541706044 | 0.0066418030155137074
optimization step | 283 | -8.399494011509395 | 0.005489010023444614
optimization step | 284 | -8.417630785355291 | 0.007914861977148942
optimization step | 285 | -8.37816454078575 | 0.010228079556889594
optimization step | 286 | -8.40394260871508 | 0.011708441787535561
optimization step | 287 | -8.392204308800933 | 0.010669526661348449
optimization step | 288 | -8.394470083732314 | 0.016657023591981148
optimization step | 289 | -8.377929410418496 | 0.011997993704242049
optimization step | 290 | -8.39463037946654 | 0.015217972044217876
optimization step | 291 | -8.405470109238447 | 0.008865559684407371
optimization step | 292 | -8.387390978287513 | 0.010455653815476178
optimization step | 293 | -8.378207061439129 | 0.013151775816727855
optimization step | 294 | -8.397427184601526 | 0.006933103899857349
optimization step | 295 | -8.36131667185746 | 0.010219380630228134
optimization step | 296 | -8.411121719341692 | 0.008265087037187717
optimization step | 297 | -8.381884582972308 | 0.015201633633635513
optimization step | 298 | -8.384572047286015 | 0.011610608129009464
optimization step | 299 | -8.407094389105769 | 0.0156644654946456
optimization step | 300 | -8.411007420675293 | 0.016921985320338603
optimization step | 301 | -8.391479514758169 | 0.008330540206370782
optimization step | 302 | -8.396241510851967 | 0.005259564800870249
optimization step | 303 | -8.406144951945262 | 0.011868826239917344
optimization step | 304 | -8.386908650325234 | 0.016228682773754683
optimization step | 305 | -8.424266808284544 | 0.018561196813413436
optimization step | 306 | -8.415615336655579 | 0.01276146682943018
optimization step | 307 | -8.398017774457424 | 0.017086322906581214
optimization step | 308 | -8.39212400399024 | 0.006228977308724544
optimization step | 309 | -8.387303718276133 | 0.013005667826788515
optimization step | 310 | -8.394339323466294 | 0.010560375338532043
optimization step | 311 | -8.412201826897764 | 0.015250125719071686
optimization step | 312 | -8.388351205591796 | 0.0051796680162271335
optimization step | 313 | -8.379895706757713 | 0.009836891184455007
optimization step | 314 | -8.391398020046019 | 0.004892745240423065
optimization step | 315 | -8.413938550649098 | 0.01175348412527822
optimization step | 316 | -8.4106851745514 | 0.013847570127864467
optimization step | 317 | -8.407965417629317 | 0.004950570609235114
optimization step | 318 | -8.38326783724202 | 0.015021067882142126
optimization step | 319 | -8.398661634002284 | 0.0073204865962557715
optimization step | 320 | -8.407724568666993 | 0.020733243541879935
optimization step | 321 | -8.394825125739597 | 0.008571768811070413
optimization step | 322 | -8.404272767665855 | 0.007457872253274869
optimization step | 323 | -8.38992793598258 | 0.009002263547900872
optimization step | 324 | -8.396991368210541 | 0.015389248682568519
optimization step | 325 | -8.396841307726252 | 0.011211744238353154
optimization step | 326 | -8.395974886982556 | 0.010942763921039041
optimization step | 327 | -8.37218433227631 | 0.01104784229577441
optimization step | 328 | -8.380437473513808 | 0.008834456728160295
optimization step | 329 | -8.373616581168703 | 0.010304742101153336
optimization step | 330 | -8.396295329917745 | 0.020074534951223914
optimization step | 331 | -8.386055378544214 | 0.010441947173286264
optimization step | 332 | -8.394569351074033 | 0.010171649109026666
optimization step | 333 | -8.390252988790056 | 0.011362596558246289
optimization step | 334 | -8.419921856012746 | 0.013959272950544352
optimization step | 335 | -8.413954874101346 | 0.010725536198313967
optimization step | 336 | -8.406800058405343 | 0.014791886156934676
optimization step | 337 | -8.404060798236689 | 0.014921116275556366
optimization step | 338 | -8.413821255901038 | 0.009552634052041544
optimization step | 339 | -8.414325313235285 | 0.016433368399637237
optimization step | 340 | -8.417047861818931 | 0.01255682016186943
optimization step | 341 | -8.40873569508475 | 0.00622889216198762
optimization step | 342 | -8.407126996364841 | 0.015713608400532396
optimization step | 343 | -8.408142767580618 | 0.012751538107626774
optimization step | 344 | -8.395031130382003 | 0.013702574669605309
optimization step | 345 | -8.398689260033004 | 0.014190922025643085
optimization step | 346 | -8.411040593221706 | 0.004767328038679253
optimization step | 347 | -8.414664241490579 | 0.003342497084909411
optimization step | 348 | -8.386654828194962 | 0.008206328390732138
optimization step | 349 | -8.411530996797605 | 0.008885788572942848
optimization step | 350 | -8.402206242790523 | 0.010285187664501923
optimization step | 351 | -8.408450400333361 | 0.0198772079620234
optimization step | 352 | -8.388770230900123 | 0.016714897744443003
optimization step | 353 | -8.400692629182505 | 0.01048194137594155
optimization step | 354 | -8.392295738783863 | 0.00821680816484182
optimization step | 355 | -8.381856974779462 | 0.013577951297174092
optimization step | 356 | -8.398159494270157 | 0.006869371837062841
optimization step | 357 | -8.393018069286821 | 0.002472179303176596
optimization step | 358 | -8.405588130026484 | 0.00720864706885272
optimization step | 359 | -8.408341305837599 | 0.010025259343160594
optimization step | 360 | -8.414856089686314 | 0.008911003112272418
optimization step | 361 | -8.401722120274993 | 0.01830250691910063
optimization step | 362 | -8.404343395867272 | 0.008846541600372078
optimization step | 363 | -8.41309418632353 | 0.006198882199405017
optimization step | 364 | -8.404361780686306 | 0.010963058238919769
optimization step | 365 | -8.383374326035687 | 0.011723169021220994
optimization step | 366 | -8.393615772180961 | 0.00650073875351934
optimization step | 367 | -8.40729980530911 | 0.009218421407125782
optimization step | 368 | -8.397153925348508 | 0.004051906095899937
optimization step | 369 | -8.408667998008378 | 0.010961108447027716
optimization step | 370 | -8.372453961396785 | 0.006712942237068985
optimization step | 371 | -8.409964222294457 | 0.006322278286514255
optimization step | 372 | -8.397741322561798 | 0.01543849812150425
optimization step | 373 | -8.399649927126791 | 0.014374414389649853
optimization step | 374 | -8.394716732635962 | 0.009581721502060816
optimization step | 375 | -8.402206849113824 | 0.015652733457946676
optimization step | 376 | -8.406542931956071 | 0.008351077592507871
optimization step | 377 | -8.431869781045972 | 0.012436798882509923
optimization step | 378 | -8.389512254494297 | 0.008097477494686355
optimization step | 379 | -8.404258770014547 | 0.017977905068905328
optimization step | 380 | -8.390070919290025 | 0.010281756850769005
optimization step | 381 | -8.39030365556523 | 0.013375532112435034
optimization step | 382 | -8.363676081607938 | 0.014650875302142463
optimization step | 383 | -8.4004408069908 | 0.008214916612375115
optimization step | 384 | -8.38276571958918 | 0.0025835947221547737
optimization step | 385 | -8.402241996758608 | 0.010837240460531266
optimization step | 386 | -8.390146157344585 | 0.009202667827428982
optimization step | 387 | -8.393982842664052 | 0.012367137021768516
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0001,
        "n_blocks": 10,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 10,
        "n_optimization_steps": 500,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+1)
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -8.395363995558018 | 0.011117354042172106
optimization step | 1 | -8.39901285385298 | 0.008000723663850228
optimization step | 2 | -8.395044251723842 | 0.004689417683808768
optimization step | 3 | -8.415471832061254 | 0.007144593425010929
optimization step | 4 | -8.400486402904193 | 0.009814914117281838
optimization step | 5 | -8.394994805404487 | 0.007998377125543096
optimization step | 6 | -8.40846186921037 | 0.005730467726450832
optimization step | 7 | -8.400528612255998 | 0.006298275896439315
optimization step | 8 | -8.39364331612106 | 0.0071921303396390536
optimization step | 9 | -8.3860796543927 | 0.007530182867138234
optimization step | 10 | -8.406501638529573 | 0.010300802105753747
optimization step | 11 | -8.409159474594347 | 0.009029704975028158
optimization step | 12 | -8.398698967972829 | 0.008871018272118855
optimization step | 13 | -8.394998176696161 | 0.007484091770074912
optimization step | 14 | -8.399287080979274 | 0.007645212355094298
optimization step | 15 | -8.396038704471144 | 0.006423645778683616
optimization step | 16 | -8.40016564923489 | 0.004067149754264194
optimization step | 17 | -8.404493347210181 | 0.006694434728839653
optimization step | 18 | -8.388111840108852 | 0.008419728552820648
optimization step | 19 | -8.396361487027798 | 0.007944080407764075
optimization step | 20 | -8.397660591841142 | 0.0065210208941718206
optimization step | 21 | -8.403012562891444 | 0.007133132330198561
optimization step | 22 | -8.398190880529109 | 0.008369706726700645
optimization step | 23 | -8.417297518308335 | 0.007569992888232473
optimization step | 24 | -8.383984938898022 | 0.008071420419714719
optimization step | 25 | -8.399552801692826 | 0.005946894302278187
optimization step | 26 | -8.40253831627056 | 0.006320721118119962
optimization step | 27 | -8.398582473180166 | 0.00957626335783983
optimization step | 28 | -8.398465221574687 | 0.006843570130961332
optimization step | 29 | -8.406122770180506 | 0.00601156888860752
optimization step | 30 | -8.3968670826479 | 0.0071584981303194056
optimization step | 31 | -8.40119521803957 | 0.005199167850678682
optimization step | 32 | -8.387017036000884 | 0.007947733834447761
optimization step | 33 | -8.39863588040521 | 0.007891867092319826
optimization step | 34 | -8.392080495061794 | 0.0053436990960081835
optimization step | 35 | -8.402028130431571 | 0.009987956040721037
optimization step | 36 | -8.39188709976941 | 0.008102594495510332
optimization step | 37 | -8.39930554885659 | 0.008767585185911659
optimization step | 38 | -8.397896947346009 | 0.00574979563387091
optimization step | 39 | -8.39868897703382 | 0.008572451583572793
optimization step | 40 | -8.402317938954972 | 0.006168889278761282
optimization step | 41 | -8.39963615420612 | 0.009931726748799023
optimization step | 42 | -8.40116435920427 | 0.010034162194210416
optimization step | 43 | -8.396375686075608 | 0.0074270988680316665
optimization step | 44 | -8.413462665462783 | 0.008478553650578048
optimization step | 45 | -8.388801069657294 | 0.009155068852786049
optimization step | 46 | -8.399900446010957 | 0.005293668203354689
optimization step | 47 | -8.404629891757612 | 0.006110035672739414
optimization step | 48 | -8.403769874285729 | 0.010319122536124832
optimization step | 49 | -8.406164253564633 | 0.007094367739442103
optimization step | 50 | -8.399428729733446 | 0.004590952814659362
optimization step | 51 | -8.399350629812414 | 0.008384657350871832
optimization step | 52 | -8.40004980643041 | 0.011061336285238064
optimization step | 53 | -8.410106858297087 | 0.0061724014085039
optimization step | 54 | -8.408278980037062 | 0.006546856969927326
optimization step | 55 | -8.403880179405707 | 0.006843035220122694
optimization step | 56 | -8.413705437730382 | 0.005547711746861678
optimization step | 57 | -8.40905110829894 | 0.0077749339301660895
optimization step | 58 | -8.413984487860079 | 0.011188483584358174
optimization step | 59 | -8.40488483588226 | 0.008345894533457568
optimization step | 60 | -8.399112851912353 | 0.006065125628813234
optimization step | 61 | -8.412055361926747 | 0.00970725742506443
optimization step | 62 | -8.395936932390935 | 0.006235862842044687
optimization step | 63 | -8.403583411774509 | 0.008555217092649664
optimization step | 64 | -8.3996050264486 | 0.005978482752751748
optimization step | 65 | -8.410135267302685 | 0.006021984356749928
optimization step | 66 | -8.392703716626553 | 0.003511767778921567
optimization step | 67 | -8.407198121761104 | 0.007790064790169353
optimization step | 68 | -8.407528417901398 | 0.005915353468553158
optimization step | 69 | -8.410476407830073 | 0.006875194433255767
optimization step | 70 | -8.40721565312315 | 0.0067713212021922676
optimization step | 71 | -8.40847077341033 | 0.009239044543374752
optimization step | 72 | -8.407107634208023 | 0.008572584106410633
optimization step | 73 | -8.386614941217731 | 0.00965026265699544
optimization step | 74 | -8.39719188196384 | 0.008622034852456297
optimization step | 75 | -8.404847319234358 | 0.007777009525582683
optimization step | 76 | -8.422743068074386 | 0.007956495070474207
optimization step | 77 | -8.4048640541257 | 0.006344677381101994
optimization step | 78 | -8.3937225974633 | 0.009089531093859087
optimization step | 79 | -8.415977332200715 | 0.00918898947293697
optimization step | 80 | -8.407900018971542 | 0.009034903269487974
optimization step | 81 | -8.397041398203228 | 0.008806423511245663
optimization step | 82 | -8.410252863206676 | 0.005824953892227857
optimization step | 83 | -8.409629126491657 | 0.005921051269271504
optimization step | 84 | -8.414875777101193 | 0.006215983953374726
optimization step | 85 | -8.411825238941692 | 0.00692267509204987
optimization step | 86 | -8.402138872359536 | 0.006480933480955953
optimization step | 87 | -8.40809838347366 | 0.004024056328941834
optimization step | 88 | -8.41636641044525 | 0.006028393915912457
optimization step | 89 | -8.412907398105457 | 0.007170978748982121
optimization step | 90 | -8.413802006833423 | 0.006662331959366289
optimization step | 91 | -8.41293911328253 | 0.006335425379220031
optimization step | 92 | -8.410011690113137 | 0.00584089780318844
optimization step | 93 | -8.395206644956632 | 0.0046433299481508485
optimization step | 94 | -8.39369433910522 | 0.008377474020130374
optimization step | 95 | -8.401237307428488 | 0.004112040129733625
optimization step | 96 | -8.414177908479672 | 0.009360841484209964
optimization step | 97 | -8.39184005206505 | 0.008409516697151054
optimization step | 98 | -8.39205398722899 | 0.006716963580319225
optimization step | 99 | -8.40527429570509 | 0.006898394330252395
optimization step | 100 | -8.400789787331394 | 0.006754917977868722
optimization step | 101 | -8.390539366676403 | 0.005386038061421335
optimization step | 102 | -8.410552883915631 | 0.005317923475166441
optimization step | 103 | -8.407465043506352 | 0.004803226087762725
optimization step | 104 | -8.402534058408008 | 0.010562762797013198
optimization step | 105 | -8.427257675372676 | 0.006984170916093765
optimization step | 106 | -8.417690861011149 | 0.0052640515840153244
optimization step | 107 | -8.408141386651067 | 0.007671812245757896
optimization step | 108 | -8.420698182946207 | 0.0065291385120159725
optimization step | 109 | -8.401185270779576 | 0.008683641556092608
optimization step | 110 | -8.408428656708631 | 0.005716025630742082
optimization step | 111 | -8.407749540273198 | 0.00519679511345473
optimization step | 112 | -8.406465055648855 | 0.008615583404590565
optimization step | 113 | -8.40864790312403 | 0.011340482109149801
optimization step | 114 | -8.403313806606258 | 0.007866658049668376
optimization step | 115 | -8.414520968286993 | 0.007409794753592664
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0001,
        "n_blocks": 10,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 10,
        "n_optimization_steps": 150,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma",
            "tau",
            "sigma_tau"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+tau+sigma_tau+1)
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -8.44952553023411 | 0.00968268733069298
optimization step | 1 | -8.445213816551055 | 0.00476545458708387
optimization step | 2 | -8.444296178765038 | 0.008513935674773267
optimization step | 3 | -8.461983885645111 | 0.006831179151666451
optimization step | 4 | -8.44896646902494 | 0.008801532780968022
optimization step | 5 | -8.429593582415299 | 0.0056086511055214
optimization step | 6 | -8.459278200660993 | 0.00774085130855283
optimization step | 7 | -8.459815279778798 | 0.00768501528006853
optimization step | 8 | -8.445845174040983 | 0.006484168172006954
optimization step | 9 | -8.440307573095122 | 0.00877605604644812
optimization step | 10 | -8.44792135999495 | 0.008681969259978014
optimization step | 11 | -8.449027974565578 | 0.006032498062416359
optimization step | 12 | -8.455288065094553 | 0.007359361616439667
optimization step | 13 | -8.453892000430235 | 0.005460001145178748
optimization step | 14 | -8.455898593532346 | 0.006406448237196148
optimization step | 15 | -8.44416989266537 | 0.005512033196243083
optimization step | 16 | -8.451589659628814 | 0.006539603209319288
optimization step | 17 | -8.452616092889304 | 0.004276810775656309
optimization step | 18 | -8.439049836110385 | 0.008562992546728057
optimization step | 19 | -8.450550796371488 | 0.009298051790970567
optimization step | 20 | -8.453491785592316 | 0.008027269635331069
optimization step | 21 | -8.450951433264363 | 0.004497681397253409
optimization step | 22 | -8.443877731950092 | 0.007571882140165713
optimization step | 23 | -8.458163306005238 | 0.007383058774407623
optimization step | 24 | -8.435321773380029 | 0.009080066766754196
optimization step | 25 | -8.448790821299749 | 0.006886625727688885
optimization step | 26 | -8.44756987689235 | 0.0047477321082865275
optimization step | 27 | -8.447462886168092 | 0.00857767224331558
optimization step | 28 | -8.449819320261009 | 0.0056487854721759875
optimization step | 29 | -8.456995444943859 | 0.006740213845635672
optimization step | 30 | -8.447902547776339 | 0.0056139251200079425
optimization step | 31 | -8.450313349159552 | 0.006784729501973051
optimization step | 32 | -8.43582686180531 | 0.008683299027805461
optimization step | 33 | -8.444157159184378 | 0.007379710544712563
optimization step | 34 | -8.443735113630453 | 0.006700995361301433
optimization step | 35 | -8.455996612163405 | 0.00651012784106143
optimization step | 36 | -8.442096315577233 | 0.009442710868203027
optimization step | 37 | -8.45397617311865 | 0.004682178962926293
optimization step | 38 | -8.451512343238011 | 0.006085528269120676
optimization step | 39 | -8.452173544365635 | 0.006193138055042807
optimization step | 40 | -8.45297789467196 | 0.004165274520868429
optimization step | 41 | -8.447057630426162 | 0.0063684532167976515
optimization step | 42 | -8.453816060072855 | 0.006925822738627367
optimization step | 43 | -8.442989734932036 | 0.0069823192900442645
optimization step | 44 | -8.454515446987427 | 0.004780638162043181
optimization step | 45 | -8.445193650948056 | 0.0065416780942648784
optimization step | 46 | -8.452086546899645 | 0.0069642262052165985
optimization step | 47 | -8.462367897723981 | 0.004694256409913629
optimization step | 48 | -8.453645735006575 | 0.0057383144590040834
optimization step | 49 | -8.455919094422578 | 0.00789710784317859
optimization step | 50 | -8.449533353881304 | 0.005649215968561423
optimization step | 51 | -8.442995095431321 | 0.005745228306202885
optimization step | 52 | -8.45605842360377 | 0.00792727426818105
optimization step | 53 | -8.451977583500982 | 0.00678971166928452
optimization step | 54 | -8.45577230734717 | 0.007154204983251807
optimization step | 55 | -8.451445845527251 | 0.008629300895009118
optimization step | 56 | -8.461884064178289 | 0.0057176930316349475
optimization step | 57 | -8.45804526014704 | 0.005145912295645246
optimization step | 58 | -8.45829425980337 | 0.004700524688011893
optimization step | 59 | -8.458335983949643 | 0.009598895295214053
optimization step | 60 | -8.445312062792546 | 0.005912616889318167
optimization step | 61 | -8.45432194318476 | 0.008343961980646028
optimization step | 62 | -8.443568027644705 | 0.006307081556513048
optimization step | 63 | -8.459017795242408 | 0.006585110677590229
optimization step | 64 | -8.444301786693071 | 0.009194912658036701
optimization step | 65 | -8.462299062098628 | 0.006760236840756908
optimization step | 66 | -8.451575443333946 | 0.005010395252064932
optimization step | 67 | -8.448391198918037 | 0.004564267841978363
optimization step | 68 | -8.455445065843715 | 0.009274146638664639
optimization step | 69 | -8.458948933052536 | 0.006614951439061056
optimization step | 70 | -8.455739408814734 | 0.005775890924368953
optimization step | 71 | -8.450297711433393 | 0.0060763370571172565
optimization step | 72 | -8.449325065932467 | 0.009577971169953564
optimization step | 73 | -8.441158042873273 | 0.004800192946221825
optimization step | 74 | -8.445374232578112 | 0.006028364314003188
optimization step | 75 | -8.444896599863647 | 0.005481511661907373
optimization step | 76 | -8.46376815060167 | 0.006457682783029119
optimization step | 77 | -8.453990042050219 | 0.008389250074697678
optimization step | 78 | -8.443709051873519 | 0.008397173732127377
optimization step | 79 | -8.459939701496976 | 0.005186521718274619
optimization step | 80 | -8.454922077384598 | 0.005896399966645742
optimization step | 81 | -8.447168821951351 | 0.008058481200423748
optimization step | 82 | -8.453284090631513 | 0.005864653708590155
optimization step | 83 | -8.455458256513316 | 0.005141247672009506
optimization step | 84 | -8.455818486408763 | 0.007028836797415909
optimization step | 85 | -8.455426835688915 | 0.00585913003181854
optimization step | 86 | -8.436458641582094 | 0.005448212933348904
optimization step | 87 | -8.460332792347046 | 0.005261246835255421
optimization step | 88 | -8.450371319190715 | 0.007266247885404815
optimization step | 89 | -8.454110302730697 | 0.007014083733969957
optimization step | 90 | -8.457477924119432 | 0.006049919997975911
optimization step | 91 | -8.454097934840238 | 0.007880301860383387
optimization step | 92 | -8.45361813616882 | 0.004960315903100086
optimization step | 93 | -8.447939288614041 | 0.007090443470900615
optimization step | 94 | -8.438023251588675 | 0.006317569010683106
optimization step | 95 | -8.451183018181103 | 0.002286071090902833
optimization step | 96 | -8.45506199497238 | 0.006795769714358381
optimization step | 97 | -8.449000962458818 | 0.0067761560636948655
optimization step | 98 | -8.44695794810773 | 0.005923057095909782
optimization step | 99 | -8.454271781839337 | 0.004278484280936465
optimization step | 100 | -8.458248014557233 | 0.004860704480306468
optimization step | 101 | -8.439008552226312 | 0.005043797239533712
optimization step | 102 | -8.45249719667939 | 0.007329315835717755
optimization step | 103 | -8.449162247681608 | 0.005390974223484634
optimization step | 104 | -8.450379028815728 | 0.00788611335987895
optimization step | 105 | -8.464354438014476 | 0.008112928974031198
optimization step | 106 | -8.460729804309363 | 0.006846789868605538
optimization step | 107 | -8.458021951529819 | 0.007414901296840155
optimization step | 108 | -8.460147209619034 | 0.008888206986951237
optimization step | 109 | -8.449951696026101 | 0.006769982334298484
optimization step | 110 | -8.464780437334728 | 0.005077793550168008
optimization step | 111 | -8.453330289903793 | 0.0060850046604969035
optimization step | 112 | -8.451922839472394 | 0.003763447730203848
optimization step | 113 | -8.461120351498657 | 0.00961980401957761
optimization step | 114 | -8.458722082780216 | 0.004877218083420221
optimization step | 115 | -8.456449502015316 | 0.007665249051601494
optimization step | 116 | -8.454235671454501 | 0.0070651450182929315
optimization step | 117 | -8.450081844472138 | 0.006031730513402058
optimization step | 118 | -8.457797246783443 | 0.009997896558233367
optimization step | 119 | -8.456781672667782 | 0.005825540771079091
optimization step | 120 | -8.445752626676311 | 0.007581734878663079
optimization step | 121 | -8.458345715904596 | 0.00739325140464348
optimization step | 122 | -8.448839849609316 | 0.007416040866449435
optimization step | 123 | -8.442612574549962 | 0.009252682680010636
optimization step | 124 | -8.44009443738551 | 0.0073612466964953745
optimization step | 125 | -8.445680475696248 | 0.007368928300657078
optimization step | 126 | -8.463164826622265 | 0.007533725590802226
optimization step | 127 | -8.450378014848539 | 0.007489460959550859
optimization step | 128 | -8.453813260315021 | 0.008137881633108983
optimization step | 129 | -8.45479268915359 | 0.003341796228370475
optimization step | 130 | -8.453273643215482 | 0.006729505118516333
optimization step | 131 | -8.45026042855128 | 0.006407389895894421
optimization step | 132 | -8.4484519987433 | 0.006554299946107119
optimization step | 133 | -8.451806982461253 | 0.003944360074660058
optimization step | 134 | -8.460457796043734 | 0.007546688839194873
optimization step | 135 | -8.465902872951684 | 0.005551616934617892
optimization step | 136 | -8.450483600391998 | 0.0062622625744530105
optimization step | 137 | -8.461846395923043 | 0.007039249883923759
optimization step | 138 | -8.445949879993355 | 0.004217210007346429
optimization step | 139 | -8.452554278358015 | 0.005827531834788846
optimization step | 140 | -8.44689412590039 | 0.006343334804618996
optimization step | 141 | -8.45609975442368 | 0.006177756582164507
optimization step | 142 | -8.443052008938675 | 0.004752763930316916
optimization step | 143 | -8.445150675074844 | 0.006010882084779391
optimization step | 144 | -8.44570023611428 | 0.005931919181883449
optimization step | 145 | -8.457379568080361 | 0.005783131650935191
optimization step | 146 | -8.452264144498972 | 0.004519752338068184
optimization step | 147 | -8.456268613523473 | 0.005497719935884907
optimization step | 148 | -8.448027070654241 | 0.005340726640043005
optimization step | 149 | -8.459379147071854 | 0.0043774271474167365
# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 2,
    "n_proton": 1,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0001,
        "n_blocks": 10,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 10,
        "n_optimization_steps": 350,
        "n_void_steps": 200,
        "n_walkers": 2000,
        "print_local_energy": true,
        "seed": 0,
        "walker_step_size": 0.2
    },
    "potential_energy": "arxiv_2102_02327v1",
    "potential_kwargs": {
        "R3": 1.5,
        "model_string": "o"
    },
    "wave_function": {
        "jastro_list": [
            "2b",
            "3b",
            "sigma",
            "tau",
            "sigma_tau"
        ],
        "n_dense": 6,
        "n_hidden_layers": 2,
        "seed": 0,
        "wave_function_file": "wave_function_parameters_0.npy"
    }
}
```
## Building Wave Function System
Wave Function Expression: 2b*3b*(sigma+tau+sigma_tau+1)
## Wave Function Parameters
reading wave function parameters from: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/a_3/3_body_spin_new_potential/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | -8.453439734386519 | 0.008224263836573516
optimization step | 1 | -8.447940800930757 | 0.005869054919223317
optimization step | 2 | -8.451751129513292 | 0.005545193912450165
optimization step | 3 | -8.461963617968198 | 0.00531831495302524
optimization step | 4 | -8.45525102919803 | 0.007552194208643584
optimization step | 5 | -8.43576464703177 | 0.006025789779420929
optimization step | 6 | -8.459484322127 | 0.005026706876865264
optimization step | 7 | -8.453314482525306 | 0.006021528761279168
optimization step | 8 | -8.442046829958434 | 0.005021033285071202
optimization step | 9 | -8.444464895551388 | 0.00698851428240695
optimization step | 10 | -8.458975242733818 | 0.007861689043222663
optimization step | 11 | -8.45846871646795 | 0.0069654495190536875
optimization step | 12 | -8.457658210462409 | 0.006423757682639343
optimization step | 13 | -8.45144982641728 | 0.0036338895201586886
optimization step | 14 | -8.4542764796906 | 0.009047358505337801
optimization step | 15 | -8.445260092272836 | 0.007363794052948714
optimization step | 16 | -8.45354479570882 | 0.004494021531557815
optimization step | 17 | -8.451546580528788 | 0.006379541719738815
optimization step | 18 | -8.442772245477693 | 0.006680946230693076
optimization step | 19 | -8.447215640044597 | 0.0077771146147927244
optimization step | 20 | -8.460690323791502 | 0.0073046509375091564
optimization step | 21 | -8.451818671222693 | 0.007336098621891915
optimization step | 22 | -8.443936584095754 | 0.008745156324392229
optimization step | 23 | -8.458672453597863 | 0.0062555476843028195
optimization step | 24 | -8.439455548856523 | 0.004778860097359344
optimization step | 25 | -8.445524453970167 | 0.005191412799170288
optimization step | 26 | -8.448107186989649 | 0.0063710742560325224
optimization step | 27 | -8.452250303893523 | 0.006560386323679445
optimization step | 28 | -8.453069182460286 | 0.006309244182528496
optimization step | 29 | -8.456654425715326 | 0.006610167111674733
optimization step | 30 | -8.448386771786053 | 0.005664009764960823
optimization step | 31 | -8.449002571400568 | 0.005385354395151283
optimization step | 32 | -8.445621129736166 | 0.008704238080125451
optimization step | 33 | -8.451318634992022 | 0.00609838555015048
optimization step | 34 | -8.443590622287399 | 0.003833775922541245
optimization step | 35 | -8.451438503313096 | 0.006527860142916609
optimization step | 36 | -8.437707980175903 | 0.007077331975085914
optimization step | 37 | -8.452803871265377 | 0.004614147347283498
optimization step | 38 | -8.452260551072454 | 0.004704565448988513
optimization step | 39 | -8.456627651882243 | 0.00708196236348986
optimization step | 40 | -8.450760509549298 | 0.002840373732151409
optimization step | 41 | -8.450459197970849 | 0.007294949076432928
optimization step | 42 | -8.456713398578216 | 0.006196085733273346
optimization step | 43 | -8.450650010767273 | 0.005595946123267349
optimization step | 44 | -8.451624037058227 | 0.004703530869717144
optimization step | 45 | -8.444229267298295 | 0.00620349676642791
optimization step | 46 | -8.450749653354183 | 0.005570601437232679
optimization step | 47 | -8.458939450417189 | 0.005528954763593747
optimization step | 48 | -8.460364530879454 | 0.006077547476116172
optimization step | 49 | -8.464265661648225 | 0.006546569342862624
optimization step | 50 | -8.445396816674718 | 0.006706566217206326
optimization step | 51 | -8.452942388746752 | 0.006399115343613652
optimization step | 52 | -8.459948833700286 | 0.008964333990001227
optimization step | 53 | -8.446893247258439 | 0.005333966898063019
optimization step | 54 | -8.454118146588442 | 0.0059477207962412805
optimization step | 55 | -8.448483586791578 | 0.009042635582435206
optimization step | 56 | -8.461257264393222 | 0.004553069859773965
optimization step | 57 | -8.44834449290745 | 0.006208016854852138
optimization step | 58 | -8.461221332207739 | 0.004163666955235137
optimization step | 59 | -8.457256662718974 | 0.008978459768215518
optimization step | 60 | -8.449010088289384 | 0.0038823701823910875
optimization step | 61 | -8.455220511922212 | 0.009286107025939532
optimization step | 62 | -8.447102488498931 | 0.004314462916103185
optimization step | 63 | -8.446303470709779 | 0.0051370105655108075
optimization step | 64 | -8.447319388366838 | 0.0072305866003961085
optimization step | 65 | -8.459613316773344 | 0.005736737384773067
optimization step | 66 | -8.451264544270021 | 0.005424621172803347
optimization step | 67 | -8.444617088216054 | 0.006922110205075187
optimization step | 68 | -8.455643306515864 | 0.009314926401585804
optimization step | 69 | -8.45990972703005 | 0.008642452811099327
optimization step | 70 | -8.454985213591916 | 0.003890412337649148
optimization step | 71 | -8.454118781454692 | 0.007166903331379036
optimization step | 72 | -8.456695576519426 | 0.009000384264777252
optimization step | 73 | -8.447766938671581 | 0.005200900301354821
optimization step | 74 | -8.451711237029789 | 0.006531268245908315
optimization step | 75 | -8.450008010717177 | 0.007296299083042509
optimization step | 76 | -8.466235107953784 | 0.008016325608146304
optimization step | 77 | -8.448884745674766 | 0.0072527627932477004
optimization step | 78 | -8.451322022068663 | 0.00832975484184379
optimization step | 79 | -8.462972185683906 | 0.007808625565286287
optimization step | 80 | -8.452862570310597 | 0.00784661786415115
optimization step | 81 | -8.442545719883345 | 0.008145323548724496
optimization step | 82 | -8.457592781176764 | 0.006392195393784236
optimization step | 83 | -8.458335891810595 | 0.005509757886626472
optimization step | 84 | -8.460328020433614 | 0.004186502729246655
optimization step | 85 | -8.456584308775017 | 0.0057734989694792765
optimization step | 86 | -8.445461411762036 | 0.0075234611268379615
optimization step | 87 | -8.46441348212328 | 0.004837964604821181
optimization step | 88 | -8.448801803373836 | 0.003599762140565995
optimization step | 89 | -8.451127059211078 | 0.008080342287486667
optimization step | 90 | -8.458040237440974 | 0.006544715191882044
optimization step | 91 | -8.453494326436305 | 0.00763189597459127
optimization step | 92 | -8.455598366311524 | 0.003739595236465958
optimization step | 93 | -8.452378401495917 | 0.006183225079804437
optimization step | 94 | -8.443221528744912 | 0.007548923914788877
optimization step | 95 | -8.453690501987655 | 0.0039096208948567164
optimization step | 96 | -8.454017086352152 | 0.005937387817425075
optimization step | 97 | -8.446435723803553 | 0.005210281965886233
optimization step | 98 | -8.449662598170361 | 0.00555201761074064
optimization step | 99 | -8.457909335519105 | 0.0038718098672459655
optimization step | 100 | -8.458096014639361 | 0.004303361064318116
optimization step | 101 | -8.4436388012409 | 0.00547161940867688
optimization step | 102 | -8.454591294984677 | 0.005661428366297535
optimization step | 103 | -8.457898138228233 | 0.005306382281727153
optimization step | 104 | -8.455309736185376 | 0.00832248490918646
optimization step | 105 | -8.46543275712366 | 0.007805661693481493
optimization step | 106 | -8.461793640820776 | 0.006248290315617753
optimization step | 107 | -8.457919720959918 | 0.005486276048526626
optimization step | 108 | -8.455597600655834 | 0.00644970564687488
optimization step | 109 | -8.448751265039409 | 0.006688618946697147
optimization step | 110 | -8.45925560054966 | 0.0025295847809485854
optimization step | 111 | -8.448944433924417 | 0.005376952137139091
optimization step | 112 | -8.454972916598411 | 0.004328688544337389
optimization step | 113 | -8.45719104148783 | 0.008382062563075849
optimization step | 114 | -8.464781986032268 | 0.004322163451497145
optimization step | 115 | -8.45943379148258 | 0.007175754632747606
optimization step | 116 | -8.456947863395099 | 0.006400373606969744
optimization step | 117 | -8.455916440790704 | 0.006000128820898111
optimization step | 118 | -8.459444906476747 | 0.009605616478148069
optimization step | 119 | -8.464267891857906 | 0.0062086394796981685
optimization step | 120 | -8.447755379286582 | 0.005575915576285358
optimization step | 121 | -8.455003761768662 | 0.007755803037379086
optimization step | 122 | -8.450495320289363 | 0.00942673997289177
optimization step | 123 | -8.449220633691473 | 0.00758589742301412
optimization step | 124 | -8.442251977581945 | 0.004364702032240759
optimization step | 125 | -8.442271540353179 | 0.005849761833128559
optimization step | 126 | -8.464202993086467 | 0.006045059018379429
optimization step | 127 | -8.45671195609007 | 0.005219853514658031
optimization step | 128 | -8.45283743942898 | 0.01081868273415719
optimization step | 129 | -8.458381808570692 | 0.0052545050156264025
optimization step | 130 | -8.454074334248288 | 0.006857596685862535
optimization step | 131 | -8.448673460122354 | 0.007308430691730186
optimization step | 132 | -8.453027112459433 | 0.006718278687915278
optimization step | 133 | -8.451832065691674 | 0.004183650087850808
optimization step | 134 | -8.458372262196743 | 0.00785358007555042
optimization step | 135 | -8.468795099858184 | 0.007232555597185711
optimization step | 136 | -8.449638560109232 | 0.007152448230457211
optimization step | 137 | -8.462422042750402 | 0.007976376179500396
optimization step | 138 | -8.449192446497491 | 0.004266869529351584
optimization step | 139 | -8.458414315191083 | 0.006912028494083801
optimization step | 140 | -8.44235225862074 | 0.006516872631732332
optimization step | 141 | -8.45612498060722 | 0.0051580999494145775
optimization step | 142 | -8.45192011257404 | 0.0053530119619077735
optimization step | 143 | -8.447760220567709 | 0.008992339062281501
optimization step | 144 | -8.449999973265045 | 0.004647710937582221
optimization step | 145 | -8.463244844300135 | 0.005101001016634209
optimization step | 146 | -8.447804474457776 | 0.0067180622347249265
optimization step | 147 | -8.457074768611907 | 0.007128807771562103
optimization step | 148 | -8.44596212122994 | 0.0029695260281321894
optimization step | 149 | -8.461151160200343 | 0.006267889860347237
optimization step | 150 | -8.453125993389323 | 0.006999897508193941
optimization step | 151 | -8.443384836338382 | 0.005864408390533888
optimization step | 152 | -8.457968016473872 | 0.005190428103948704
optimization step | 153 | -8.44620477158804 | 0.005458016766208367
optimization step | 154 | -8.452548894709116 | 0.004203167909945568
optimization step | 155 | -8.446217040892925 | 0.009613031650940308
optimization step | 156 | -8.460054012765696 | 0.007882867588015327
optimization step | 157 | -8.446272487199343 | 0.008037366542638279
optimization step | 158 | -8.442385851832746 | 0.00552090802615139
optimization step | 159 | -8.452740080711513 | 0.00498563011989185
optimization step | 160 | -8.452877621126696 | 0.005880250781563854
optimization step | 161 | -8.461576329426217 | 0.004956175730201058
optimization step | 162 | -8.450149087555518 | 0.0048602121993715115
optimization step | 163 | -8.453435086058887 | 0.008922131617933718
optimization step | 164 | -8.458826183336608 | 0.006024164049643356
optimization step | 165 | -8.459752460735938 | 0.006070484906729615
optimization step | 166 | -8.45293664221786 | 0.006219710546282181
optimization step | 167 | -8.459666340147791 | 0.007565188948429967
optimization step | 168 | -8.449795627593161 | 0.006528488744029351
optimization step | 169 | -8.456009808105517 | 0.006956804590294154
optimization step | 170 | -8.449752012953937 | 0.008645985065954926
optimization step | 171 | -8.456684246469687 | 0.006432982291958927
optimization step | 172 | -8.44854028030856 | 0.004955817796908593
optimization step | 173 | -8.446065575847538 | 0.007591504317666655
optimization step | 174 | -8.45071729984543 | 0.005726494006481597
optimization step | 175 | -8.456753917627093 | 0.005540943657989904
optimization step | 176 | -8.45306950335554 | 0.004715415675118595
optimization step | 177 | -8.442776773872785 | 0.0057974018457526595
optimization step | 178 | -8.451527246276974 | 0.005921374021424034
optimization step | 179 | -8.452243820852638 | 0.008161170177712281
optimization step | 180 | -8.453359402165002 | 0.008088440746797276
optimization step | 181 | -8.451052889222678 | 0.003388385726042297
optimization step | 182 | -8.45523992944275 | 0.005266260771368366
optimization step | 183 | -8.460845771137361 | 0.005677919167433988
optimization step | 184 | -8.457788651360032 | 0.005085531923535657
optimization step | 185 | -8.457955988014781 | 0.005991704305755622
optimization step | 186 | -8.457740272563147 | 0.005342509730175489
optimization step | 187 | -8.456445986826317 | 0.008533205017927094
optimization step | 188 | -8.453727326997303 | 0.0040685954542432875
optimization step | 189 | -8.456186459965759 | 0.008880920027003423
optimization step | 190 | -8.451892150153004 | 0.005404106862988136
optimization step | 191 | -8.44707324298799 | 0.005978140558997853
optimization step | 192 | -8.44292861438235 | 0.006725780890645848
optimization step | 193 | -8.444594745706016 | 0.007968100510357706
optimization step | 194 | -8.45878072740549 | 0.007258116851218202
optimization step | 195 | -8.457173801939872 | 0.005401705303967497
optimization step | 196 | -8.445784937119818 | 0.005800010216980495
optimization step | 197 | -8.460482319182853 | 0.004852127089975386
optimization step | 198 | -8.45519699833554 | 0.004794059108519732
optimization step | 199 | -8.445357770968904 | 0.0072239903704667515
optimization step | 200 | -8.451645622119905 | 0.008657737246887957
optimization step | 201 | -8.445779496730996 | 0.007040873023691389
optimization step | 202 | -8.44032744604004 | 0.007987889633440421
optimization step | 203 | -8.446653624839774 | 0.0074975211054618685
optimization step | 204 | -8.453437530597236 | 0.007194750124217157
optimization step | 205 | -8.462092725032297 | 0.00721221190883257
optimization step | 206 | -8.450208094179017 | 0.009165599059939173
optimization step | 207 | -8.451784610531837 | 0.004956364075837511
optimization step | 208 | -8.454001702461932 | 0.006471835304804437
optimization step | 209 | -8.453806203029593 | 0.006567618530948393
optimization step | 210 | -8.454713581915438 | 0.005319315991867082
optimization step | 211 | -8.449784248315016 | 0.006615439619877149
optimization step | 212 | -8.46504531720188 | 0.006236664914304919
optimization step | 213 | -8.455709790266695 | 0.007400739791054123
optimization step | 214 | -8.45777252719181 | 0.006262982763809639
optimization step | 215 | -8.449538442460337 | 0.005975575615976845
optimization step | 216 | -8.459737794903178 | 0.007911028890938742
optimization step | 217 | -8.448675949227212 | 0.004919504206907267
optimization step | 218 | -8.456699717834622 | 0.008172436973999506
optimization step | 219 | -8.451404994027385 | 0.006001500199304547
optimization step | 220 | -8.453874019541969 | 0.006825423644420719
optimization step | 221 | -8.454493792706797 | 0.006196989712813959
optimization step | 222 | -8.451877131187112 | 0.008764285014165159
optimization step | 223 | -8.45164763190015 | 0.005786273445238278
optimization step | 224 | -8.451758381478367 | 0.004288910840385094
optimization step | 225 | -8.457207135562449 | 0.0061253632594261195
optimization step | 226 | -8.447847000175289 | 0.005441808778611814
optimization step | 227 | -8.447483475502846 | 0.004607816234783328
optimization step | 228 | -8.447411227701803 | 0.00857389934349292
optimization step | 229 | -8.450710194194798 | 0.006707957993685702
optimization step | 230 | -8.449058316510303 | 0.006717142342872217
optimization step | 231 | -8.435482758478255 | 0.0060300798080259186
optimization step | 232 | -8.46032777064092 | 0.005813883135559545
optimization step | 233 | -8.448149637431616 | 0.004680338441370049
optimization step | 234 | -8.45879310306214 | 0.0049251473180639375
optimization step | 235 | -8.4371316648658 | 0.007294252246414372
optimization step | 236 | -8.460974551699923 | 0.006832144669500962
optimization step | 237 | -8.460562734062567 | 0.008404581914351493
optimization step | 238 | -8.453350457072826 | 0.009538666998020721
optimization step | 239 | -8.441022531729617 | 0.007579762866156449
optimization step | 240 | -8.453987473982318 | 0.005229148383826433
optimization step | 241 | -8.454761605052592 | 0.0052086244896006936
optimization step | 242 | -8.443693255625298 | 0.006667970137971409
optimization step | 243 | -8.453217783391144 | 0.0053140722957506605
optimization step | 244 | -8.460009269142189 | 0.006896927849996584
optimization step | 245 | -8.455519129679912 | 0.005098729295541215
optimization step | 246 | -8.444463369203927 | 0.009400170895770272
optimization step | 247 | -8.453890058750373 | 0.007808372413959112
optimization step | 248 | -8.44773996976333 | 0.004824570869115817
optimization step | 249 | -8.449022622072139 | 0.005231331672366871
optimization step | 250 | -8.45001195836971 | 0.006494049626338558
optimization step | 251 | -8.452245406377198 | 0.008984823117427135
optimization step | 252 | -8.460978864248132 | 0.005411223189839956
optimization step | 253 | -8.457260117732014 | 0.009121563578917555
optimization step | 254 | -8.455926531986918 | 0.005049268474733433
optimization step | 255 | -8.456713655062602 | 0.004638686589750675
optimization step | 256 | -8.437039044177265 | 0.005609852008677571
optimization step | 257 | -8.455543603561196 | 0.0060268738513444544
optimization step | 258 | -8.45247575298669 | 0.007131476444500379
optimization step | 259 | -8.44733239698985 | 0.00961270373055826
optimization step | 260 | -8.4474719918378 | 0.004500776184778021
optimization step | 261 | -8.45777598996727 | 0.005416089129799872
optimization step | 262 | -8.454940535745171 | 0.005856009725290198
optimization step | 263 | -8.441567748346626 | 0.005380609094593031
optimization step | 264 | -8.457055983322178 | 0.0053862735615463975
optimization step | 265 | -8.451663909276487 | 0.006087516170373738
optimization step | 266 | -8.448911384428675 | 0.007423348322900025
optimization step | 267 | -8.465448841922345 | 0.005595833141612335
optimization step | 268 | -8.470001603777943 | 0.004020673435839466
optimization step | 269 | -8.448823912923308 | 0.006426177962400455
optimization step | 270 | -8.446780062966294 | 0.007484612294394101
optimization step | 271 | -8.45239194240077 | 0.0054710056050395455
optimization step | 272 | -8.447758351255548 | 0.006959092997417682
optimization step | 273 | -8.456337241666908 | 0.006007990933144839
optimization step | 274 | -8.45700450636093 | 0.00619235613784966
optimization step | 275 | -8.455102537978316 | 0.004246212456548909
optimization step | 276 | -8.459380735046963 | 0.007718343458262284
optimization step | 277 | -8.449736362336598 | 0.00607102034654324
optimization step | 278 | -8.45428703831992 | 0.004043219834534695
optimization step | 279 | -8.453352871071365 | 0.007818188364657985
optimization step | 280 | -8.456847205354626 | 0.006485886162275454
optimization step | 281 | -8.455285230259717 | 0.0050319497071075235
optimization step | 282 | -8.454083057644754 | 0.006456907372439427
optimization step | 283 | -8.451313050568524 | 0.00864816438322749
optimization step | 284 | -8.457649350445527 | 0.00759445328582343
optimization step | 285 | -8.447259246799938 | 0.006619257307004118
optimization step | 286 | -8.456128513952525 | 0.004336752973486734
optimization step | 287 | -8.463114294692454 | 0.007519826359654816
optimization step | 288 | -8.468582947766382 | 0.007226729410446387
optimization step | 289 | -8.447366104149474 | 0.009784995101293539
optimization step | 290 | -8.450044109162622 | 0.005091300392324082
optimization step | 291 | -8.46207319307436 | 0.0032209521324560036
optimization step | 292 | -8.445539521772817 | 0.008502673049997158
optimization step | 293 | -8.459535888982153 | 0.006951834481255311
optimization step | 294 | -8.45366105124299 | 0.005775702800074246
optimization step | 295 | -8.45466764701023 | 0.005048431298833889
optimization step | 296 | -8.466728482831197 | 0.005288034061194164
optimization step | 297 | -8.463351146835718 | 0.004205820926163434
optimization step | 298 | -8.442701881815744 | 0.005622127992651293
optimization step | 299 | -8.468309646913987 | 0.004990038668044673
optimization step | 300 | -8.450690000361119 | 0.004166391973435439
optimization step | 301 | -8.456439848589577 | 0.005557965141585497
optimization step | 302 | -8.449734755318593 | 0.008013531649909986
optimization step | 303 | -8.45930927448524 | 0.007302787763200287
optimization step | 304 | -8.444788916097796 | 0.006543470928529677
optimization step | 305 | -8.456381201036072 | 0.007673777384403323
optimization step | 306 | -8.45735416617294 | 0.009008675374902606
optimization step | 307 | -8.451182283202389 | 0.004951276701922656
optimization step | 308 | -8.452760468129545 | 0.008646877572954726
optimization step | 309 | -8.46734178430731 | 0.007267845202715738
optimization step | 310 | -8.460549981512832 | 0.0055495819387889845
optimization step | 311 | -8.437214760898504 | 0.007935016400362791
optimization step | 312 | -8.450173995504892 | 0.010211893290693635
optimization step | 313 | -8.456683967946551 | 0.005573601871822254
optimization step | 314 | -8.450314931672121 | 0.006589272762957327
optimization step | 315 | -8.454260549181312 | 0.006600151530876631
optimization step | 316 | -8.455499035813167 | 0.004867027896141126
optimization step | 317 | -8.454986547964312 | 0.004139082090235267
optimization step | 318 | -8.449866311415434 | 0.003077223308231163
optimization step | 319 | -8.453698279301651 | 0.005910212202813175
optimization step | 320 | -8.447871457966746 | 0.0051941952570431615
optimization step | 321 | -8.449586076086693 | 0.004139922160549182
optimization step | 322 | -8.447846209189029 | 0.0066219677833424554
optimization step | 323 | -8.453186584466957 | 0.006133172017249237
optimization step | 324 | -8.439531997911006 | 0.005181195556879304
optimization step | 325 | -8.458486697068382 | 0.006705855050740217
optimization step | 326 | -8.451818377381837 | 0.0040200213049139625
optimization step | 327 | -8.448153392162125 | 0.007454837539503628
optimization step | 328 | -8.447287170293542 | 0.0045770449144131075
optimization step | 329 | -8.449420159012499 | 0.006585787720497289
optimization step | 330 | -8.463936875799766 | 0.003986664933371285
optimization step | 331 | -8.460790089078376 | 0.004436230485143542
optimization step | 332 | -8.454533225578416 | 0.005913446479018634
optimization step | 333 | -8.456982835248962 | 0.005608827570770285
optimization step | 334 | -8.446867217135289 | 0.00799538156073716
optimization step | 335 | -8.450397058254381 | 0.007527814005492993
