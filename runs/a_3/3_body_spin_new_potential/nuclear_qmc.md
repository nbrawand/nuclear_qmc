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
