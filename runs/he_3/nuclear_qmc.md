# Nuclear QMC Run
## Log File
/gpfs/fs1/home/nbrawand/nuclear_qmc/runs/he_3/nuclear_qmc.md
## Input File
```json
{
    "n_neutron": 1,
    "n_proton": 2,
    "optimization": {
        "epsilon_sr": 0.0001,
        "initial_walker_standard_deviation": 1.0,
        "learning_rate": 0.0002,
        "n_blocks": 10,
        "n_dimensions": 3,
        "n_equilibrium_blocks": 10,
        "n_optimization_steps": 1000,
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
creating wave function parameters file: /gpfs/fs1/home/nbrawand/nuclear_qmc/runs/he_3/wave_function_parameters_0.npy
## Hamiltonian
## Optimization
Search String | Step | Energy | Error
------------- | ---- | ------ | -----
optimization step | 0 | 1.1564689734779985 | 0.19847194862984419
optimization step | 1 | 0.8540403571232872 | 0.1238049607048368
optimization step | 2 | 0.4190879800979167 | 0.16747329276404735
optimization step | 3 | -0.5105049184060514 | 0.17572159664748546
optimization step | 4 | -0.7672014320878244 | 0.11561817848800779
optimization step | 5 | -0.7660450182930749 | 0.19823278677930342
optimization step | 6 | -1.3822835135336409 | 0.12820317902474193
optimization step | 7 | -1.426573916429263 | 0.15573693464676716
optimization step | 8 | -1.773081513071692 | 0.15865423615469268
optimization step | 9 | -2.007312931976015 | 0.06878246789751423
optimization step | 10 | -2.2066081227525767 | 0.09079833426341312
optimization step | 11 | -2.323257414567805 | 0.06793118065423809
optimization step | 12 | -2.5011884622063962 | 0.08633273203581111
optimization step | 13 | -2.5887499834025536 | 0.07640842536233991
optimization step | 14 | -2.694253723218771 | 0.08233122214691999
optimization step | 15 | -2.9464397609921176 | 0.10737615762229007
optimization step | 16 | -3.145205547887783 | 0.049705997002443617
optimization step | 17 | -3.145534062223045 | 0.08707800159562011
optimization step | 18 | -3.3412743855335485 | 0.06824684341178439
optimization step | 19 | -3.4347102292278064 | 0.08348530979697756
optimization step | 20 | -3.407004719681648 | 0.04549647720478822
optimization step | 21 | -3.571688872900263 | 0.08579987779871541
optimization step | 22 | -3.61218744736392 | 0.08774120544586791
optimization step | 23 | -3.6989872261607575 | 0.03168651408653963
optimization step | 24 | -3.988710294523816 | 0.042944784865181086
optimization step | 25 | -3.8062413930553545 | 0.0601924705022765
optimization step | 26 | -3.976146919997268 | 0.03459958477832019
optimization step | 27 | -4.025582855872898 | 0.0678631049188843
optimization step | 28 | -4.1979653714619225 | 0.06118089578127774
optimization step | 29 | -4.174732740545698 | 0.06529259782896317
optimization step | 30 | -4.431730332504115 | 0.09322410407641707
optimization step | 31 | -4.227862902736787 | 0.0544405647822607
optimization step | 32 | -4.4480706739763445 | 0.08630985560810098
optimization step | 33 | -4.508705441001537 | 0.07389920831841609
optimization step | 34 | -4.469575624881872 | 0.05870049569394758
optimization step | 35 | -4.489224716055 | 0.047263627423221606
optimization step | 36 | -4.6479441166879925 | 0.08927463898146733
optimization step | 37 | -4.656552839524021 | 0.03682492811962181
optimization step | 38 | -4.712853146093197 | 0.056573777455522806
optimization step | 39 | -4.773467143599798 | 0.042057778723834684
optimization step | 40 | -4.837433039516484 | 0.0395601958788708
optimization step | 41 | -5.01371444745871 | 0.05674399944842484
optimization step | 42 | -4.868245239791122 | 0.04966672165951515
optimization step | 43 | -5.099679233232011 | 0.05527859170664441
optimization step | 44 | -5.144233307586229 | 0.030698151990415756
optimization step | 45 | -5.0036220114927525 | 0.04431361701234126
optimization step | 46 | -5.055643203593113 | 0.027172198834569655
optimization step | 47 | -5.174571484593128 | 0.055798392903160264
optimization step | 48 | -5.235604933788114 | 0.06261744144293163
optimization step | 49 | -5.36247362247246 | 0.025923113145440255
optimization step | 50 | -5.244807131900126 | 0.052606537770263584
optimization step | 51 | -5.432577566952321 | 0.03632375689665031
optimization step | 52 | -5.382704498825789 | 0.04482311347613603
optimization step | 53 | -5.434890273065128 | 0.04930987020515139
optimization step | 54 | -5.493082817592105 | 0.04090301568535854
optimization step | 55 | -5.6061412436693585 | 0.052232381664933714
optimization step | 56 | -5.549502316328567 | 0.0341003483570099
optimization step | 57 | -5.5148730036501075 | 0.03404644601401208
optimization step | 58 | -5.653640393850813 | 0.045795687107642334
optimization step | 59 | -5.598213031421851 | 0.04180537760919865
optimization step | 60 | -5.677759829040891 | 0.04853700566738599
optimization step | 61 | -5.770488433192478 | 0.04679968679439449
optimization step | 62 | -5.796206224974638 | 0.048992772433461486
optimization step | 63 | -5.759892669870548 | 0.04529322447507566
optimization step | 64 | -5.845197149236717 | 0.02499689938952707
optimization step | 65 | -5.8395724772564295 | 0.06388336936933615
optimization step | 66 | -5.922359377014169 | 0.04151684853037673
optimization step | 67 | -5.881677750650059 | 0.04863622660513559
optimization step | 68 | -5.9243666914441135 | 0.038970198918952995
optimization step | 69 | -5.97583570044302 | 0.037018351344894805
optimization step | 70 | -5.962132633706676 | 0.03924553200114471
optimization step | 71 | -6.012575973341533 | 0.04985384337471139
optimization step | 72 | -6.086244651796886 | 0.03352046067534796
optimization step | 73 | -6.048704624321727 | 0.04537204316479247
optimization step | 74 | -6.132405496002726 | 0.022000706614366157
optimization step | 75 | -6.126914930180408 | 0.03668699832102468
optimization step | 76 | -6.217519997106505 | 0.035341263606011655
optimization step | 77 | -6.197036125489928 | 0.03713328997632814
optimization step | 78 | -6.2425863534783925 | 0.04203904237170478
optimization step | 79 | -6.26849857918601 | 0.03927616354063498
