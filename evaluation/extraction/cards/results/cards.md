# 실험 카드: CUDA 포팅 및 CROWN 검증 성능 측정 연구 노트

- 카드 154개, 작성용 151개
- 작성 가능: True, 막음 0개, 경고 31개

## S01C01 [실험 조건/관측 기록] p1
CUDA 환경 확인. nvcc --version으로 CUDA 컴파일러 버전을 확인했다. CUDA compilation tools, release 13.3, V13.3.73이며 Tue_Jun__9_14:30:19_Pacific_Daylight_Time_2026에 빌드된 버전이다.


## S01C02 [구현·방법/계획·요청] p1
lirpa_forward_backward_fc_array.cpp의 CUDA 포팅 접근. CUDA는 기본적으로 C++ 언어에서 GPU 동작을 가능하게 한 것으로, C 언어와 동일하고 CUDA 동작이 필요한 코드만 따로 바꾸어도 nvcc에서 컴파일이 가능하다. 이에 따라 lirpa_forward_backward_fc_array.cpp의 기본 틀을 가져와서 병렬 연산이 필요한 부분만 kernel 함수(device, GPU에서 동작하는 코드)로 바꾸는 접근을 계획했다.


## S01C03 [문제·목표/관측 기록] p1
mat_add 포팅 및 반복 메모리 할당 문제. mat_add부터 배열을 통한 연산을 하고 있어 CUDA로 포팅해봤다. 그런데 mat_add는 한 두 번 호출하는 것이 아니고 여러 번 호출하며 메모리 할당, 해제가 여러 번 일어나 비효율적이라고 판단했다. 최적화는 나중에 진행하기로 했다.

- 한계: mat_add의 반복 호출로 인한 메모리 할당/해제 오버헤드. 최적화는 보류.

## S01C04 [측정 결과/관측 기록] p2
CUDA 포팅 최종 검증 완료. 최종 포팅, 컴파일, 실행 검증을 완료했으며 동일하게 동작함을 확인했다.


## S01C05 [문제·목표/계획·요청] p2
1차 진행 상황 보고 이후 후속 과제: 4가지 CUDA 버전. 1차 진행 상황 보고 이후 해야 할 일로 backward only mode의 재귀 버전, loop 버전, ???(기억나지 않음), iteration 버전을 희소행렬로 만든 버전의 4가지를 CUDA 프로그램으로 만들어서 돌려보는 것이 최종 목표이다.

- 한계: 3번 항목은 작성자가 기억하지 못함.

## S01C06 [실험 조건/관측 기록] p2
프루닝 적용. 프루닝(pruning)을 적용해서 파라미터 수를 93%정도 제한했다.

- 한계: 93%는 대략적인 수치로, 정확한 제한 비율은 명시되지 않음.

## S01C07 [실험 조건/관측 기록] p2
FCN 구조 및 1차 목표. 최광훈 안테나 선택 신경망은 Fully Connected 구조이며, N(0)=256, N(1)=491, N(2)=491, N(3)=16이다. 입력은 16x16(256개), 출력은 16개, 히든 계층은 2개(N(1)=491, N(2)=491)이다. 1차 목표는 crown 알고리즘의 입력으로 테스트 데이터 1개와 eps=10^-6으로 crown 알고리즘을 돌려서 16개 출력을 뽑는 것이다. 이를 위해 my_lirpa.cu에 실제 신경망을 로드(기존에는 가상 신경망 사용)하여 16개 출력을 뽑도록 하라는 지시로 이해했다.


## S01C08 [구현·방법/관측 기록] p2,3
crown_test.cu 작성 단계. 새로운 crown_test.cu를 만들어 다음 단계로 작성했다: 모델 파일 로드 시 CustomToLirpa.py를 참조해 model.bin 구조 파악, 새로운 FullyConnectedNetwork를 heap에 할당해 읽어온 신경망 로드(트러블 슈팅 참조), .Pickle 데이터셋을 CUDA로 읽기 위해 .bin 파일로 변환(테스트 단계이므로 1개의 데이터셋 (입력 256개 1세트) 만 변환), GPU에 신경망 로드(Memcpy) 및 CROWN 호출.

- 한계: 테스트 단계이므로 데이터셋은 1개(입력 256개 1세트)만 변환.

## S01C09 [측정 결과/관측 기록] p3,4
CPU vs GPU CROWN 검증 실행 시간 측정. GPU로 포팅 후 CPU 대비 속도 향상 여부를 확인하고자 CPU ONLY 계산 버전과 GPU 버전을 만들어 신경망 정방향 통과 & CROWN 알고리즘(backward) 계산 시간을 측정했다. CPU 버전은 101.237800 ms, GPU 버전은 353.057200 ms로 GPU 버전이 더 느렸다. 두 버전 모두 출력 하한/상한 및 예측값이 동일하게 나타났다.

- CPU ONLY 실행 시간: 101.237800 ms ms (eps = 1e-06, Output Dimension: 16, MODE: CPU ONLY (no GPU))
- GPU 실행 시간: 353.057200 ms ms (eps = 1e-06, Output Dimension: 16)
- 한계: GPU 버전은 최적화되지 않은 초기 포팅 상태. 단일 테스트 데이터(1개) 기준 측정.

## S01C10 [구현·방법/계획·요청] p4
AI 조언 기반 최적화 버전 시도. AI에게 시켜서 대충 최적화한 버전 my_lirpa_op_only_GPU, crown_test_op_only_GPU.cu를 만들고 실행해봤다. 아직 결과가 정리되지 않은 시도 단계이다.

- 한계: 결과 미정리. 최적화 효과 측정 없음.

## S01C11 [구현·방법/관측 기록] p4
-lcublas 컴파일 옵션 사용. 컴파일 옵션에 -lcublas를 넣었다. -l은 라이브러리를 링크하라는 컴파일러 옵션이고, cublas는 NVIDIA가 제공하는 cuBLAS 라이브러리이다. my_lirpa_op.cu에서 cublasDgemm() (GPU 최적화 행렬곱) 함수를 사용하기 때문에 이 라이브러리를 연결해줘야 한다. CUDA Toolkit 설치 시 자동으로 포함되어 있어 별도 설치는 필요 없다.


## S02C01 [측정 결과/해석] p5
GPU Optimized (cuBLAS + Pool) 모드 실행 시간 측정. cuBLAS로 matmul을 대체하고 메모리 풀을 사용하며 벡터 연산을 CPU로 옮긴 GPU 최적화 모드에서 CROWN 검증 실행 시간을 측정했다.

- elapsed_time: 229.691800 ms ms (GPU OPTIMIZED 모드, eps = 1e-06, Output Dimension 16)
- 한계: CPU보다 여전히 느려 추가 최적화 필요

## S02C02 [측정 결과/해석] p6
Hybrid (matmul=GPU, rest=CPU) 모드 실행 시간 측정. matmul(행렬곱)만 GPU로 실행하고 나머지는 CPU로 처리하는 하이브리드 프로그램을 만들어 실행 시간을 측정했다.

- elapsed_time: 201.670100 ms ms (HYBRID 모드, eps = 1e-06, Output Dimension 16)
- 한계: CPU보다 여전히 느려 추가 최적화 필요

## S02C03 [측정 결과/계획·요청] p6
v2 적용 방안 (후속 최적화 계획). GPU 최적화 및 하이브리드 모드에서도 CPU보다 느린 문제를 해결하기 위해 v2에서 적용할 최적화 방안을 계획했다.


## S03C01 [측정 결과/관측 기록] p7,8,10
HYBRID V2(crown_test_op_hybrid_v2) 실행 시간 측정. CROWN 검증기를 HYBRID V2 모드로 실행한 경과 시간 측정 결과. 캡션에 '85ms 드디어 cpu보다 빨라짐'이라고 기록되어 있다.

- Elapsed time: 85,785300 ms ms (MODE: HYBRID V2, eps = 1e-06, Output Dimension: 16)
- 경과 시간(캡션 표기): 85ms ms (HYBRID V2 실행 결과 캡션)
- 한계: 단일 실행 측정값으로 보이며, 반복 측정 통계는 원문에 없다.

## S03C02 [측정 결과/관측 기록] p7,8,10
버전별 성능 진행(v0~v4)과 주요 변경점. CPU 전용(v0)부터 HYBRID V2(v4)까지의 버전별 실행 시간과 각 버전의 주요 구현 변경을 정리한 카드. 논문에서 방법·실험 설정·결과 절에 활용할 수 있다.

- v0 실행 시간: 96ms ms (단독 CPU 처리)
- v1 실행 시간: 355ms ms (모든 병렬처리 함수를 커널 함수로 변경 (my_lirpa.cu))
- v2 실행 시간: 229ms ms (벡터 연산 CPU 처리 + 메모리 풀 + cuBLAS 도입 (crown_test_op_only_GPU.cu))
- v3 실행 시간: 200ms ms (CPU·GPU 하이브리드 + 풀 구조체 축소 (crown_test_op_hybrid.cu))
- v4(HYBRID V2) 실행 시간: 85ms ms (crown_test_op_hybrid_v2)
- 한계: 각 버전의 측정 조건(반복 횟수, 입력 크기, 장치)이 모두 동일한지는 원문에 명시되어 있지 않다.

## S03C03 [측정 결과/관측 기록] p9
only_GPU → hybrid 전환 시 GPU→CPU 회귀 함수 목록과 풀 구조체 축소. only_GPU 버전에서 GPU로 처리하던 함수 중 hybrid 버전에서 CPU로 되돌린 함수들과 GPU 풀 구조체 축소 내용을 정리한 카드. 논문의 방법·고찰 절에서 '어떤 연산을 GPU에서 CPU로 옮겼는지'를 설명할 때 사용할 수 있다.

- 한계: 각 함수의 실제 실행 시간 기여도는 원문에 분리되어 있지 않다.

## S03C04 [측정 결과/관측 기록] p7
적용하지 않은 최적화 방안 (cuSPARSE, memcpy 제거, CUDA stream). 검토했으나 적용하지 않거나 보류한 최적화 방안들. cuSPARSE는 실제 적용해 봤으나 cuBLAS보다 느렸고, memcpy 제거와 CUDA stream은 보류되었다. 논문의 고찰·한계 절에서 '시도했으나 채택하지 않은 방향'으로 활용할 수 있다.

- 한계: cuSPARSE 테스트의 구체적 수치(절대 시간)는 원문에 없다.

## S03C05 [측정 결과/관측 기록] p10
v4(HYBRID V2) 적용 방안: cuBLAS 예열, 가중치 사전 업로드, 부분 복사. v4에서 적용한 세 가지 최적화 방안. cuBLAS 초기화를 앞당기고, 고정 가중치(W)를 GPU에 미리 올려 호출 시 복사 비용을 없애며, 512×512(2MB) 전체 복사 대신 필요한 부분만 복사하는 내용이다. 논문의 방법 절에서 최적화 기법으로 서술할 수 있다.

- 한계: 세 방안의 개별 효과(A/B 테스트)는 원문에 없으며, 종합 결과(85ms, p7:e55)는 버전별 성능 진행 카드에서 다룬다.

## S03C06 [측정 결과/관측 기록] p8,10
메모리 풀 크기 산정 근거 (Matrix 3, Vector 6 동시 사용). GPU 메모리 풀의 크기를 산정한 근거. mat_add·matmul 등에서 최대 3개 행렬(A, B, C)이 동시 사용되고, relu_relax에서 6개 벡터(lower, upper, alpha_l, beta_l, alpha_u, beta_u)가 동시 사용된다는 점을 근거로 한다. 신경망 모델 사이즈의 약 3배 메모리 점유 가능성(p8:e65)도 함께 고려된다. 논문의 방법 절에서 메모리 풀 설계 근거로 활용할 수 있다.

- 한계: 풀의 구체적 크기(바이트 단위)는 원문에 없다.

## S03C07 [측정 결과/관측 기록] p10
중간 실험 v1-1·v1-1-1과 작성자의 방법론 반성. v1-1(벡터 연산 3가지를 CPU로 회귀, 300ms)과 v1-1-1(메모리 풀만 도입, 205~250ms) 실험 및 작성자의 방법론 반성 기록. v1-1은 메모리 풀·cuBLAS 없이 CPU로 함수를 돌려서 결국 v0와 같아지는 flawed한 비교였다는 자각이 담겨 있다. 논문의 고찰 절에서 '공정한 비교를 위해 GPU 초기화 상태를 통제한 뒤 CPU 회귀를 비교해야 한다'는 교훈으로 활용할 수 있다.

- v1-1 실행 시간: 300ms ms (my_lirpa_vector_CPU (vec_add, sub, mat CPU 회귀, 메모리 풀·cuBLAS 미도입 상태))
- v1-1-1 실행 시간: 205 ~ 250ms ms (my_lirpa_mempool.cu (메모리 풀만 도입))
- 한계: v1-1·v1-1-1은 최종 비교용 실험이 아니라 중간 탐색 실험이며, 작성자도 방법론적 한계를 스스로 지적했다.

## S04C01 [측정 결과/관측 기록] p11
CROWN 검증 원본 실행 시간 (Baseline). 수정 전 원본 CROWN 검증 실행 결과. eps=1e-06, Output Dimension 16 기준 elapsed time 228,585200 ms 기록.

- elapsed_time: 228,585200 ms ms (원본 CROWN 검증 실행, eps=1e-06, Output Dimension: 16, Lower/Upper Bound 및 Normal Pred(y) 16개 출력 포함)

## S04C02 [측정 결과/관측 기록] p11,12,13
v1-1-2: 가중치 GPU 사전 업로드 실험 및 GPU warmup 발견. 가중치를 타이머 시작 전에 GPU에 사전 업로드하는 실험을 수행. 측정 시간은 149~155ms 범위로 나타났으나, 이후 작성자는 이 감소가 preload 자체가 아니라 GPU warmup 효과 때문임을 깨달음. pre_upload_weights_to_gpu가 t_start 이전에 cudaMalloc/cudaMemcpy를 호출하면서 CUDA 컨텍스트 초기화, 클럭 부스팅, JIT 컴파일 등 약 50ms의 초기화 비용이 타이머 구간 밖으로 빠져나간 것이 원인.

- elapsed_time: 149,820400 ms ms (MODE: MEMPOOL + Weights PRELOAD ON GPU, eps=1e-06, Output Dimension: 16, crown_test_mempool_preload_on_gpu_plus_bias.cu 실행 결과 (bias 포함))
- elapsed_time_range: 149~155ms ms (가중치 사전 업로드 버전(my_lirpa_mempool_preload_on_gpu.cu) 측정 시간 범위)
- 한계: v1-1-2의 시간 감소는 preload의 실제 효과가 아니라 GPU warmup에 의한 착시임을 나중에 자각. preload된 d_W 가중치를 matmul 함수에서 실제로 활용하지 않았음.

## S04C03 [측정 결과/관측 기록] p12
v1-1-2: bias까지 GPU 사전 업로드 시도. bias 벡터까지 GPU에 사전 업로드해 보았으나 성능 개선에 큰 의미가 없었음. bias는 Vector로 약 4KB 정도 크기라 CPU가 연산하기에도 무리가 없는 수준이기 때문으로 해석.


## S04C04 [측정 결과/관측 기록] p14
v1-1-3: preload fix - 미리 로드한 가중치 제대로 활용 실험. v1-1-2에서 preload한 가중치를 실제로 활용하지 않았던 문제를 수정하여, double* 대신 Matrix* 타입을 사용하고 pre_upload_weights_to_gpu를 Matrix 할당/복사로 변경. 그러나 my_lirpa_mempool_preload_on_gpu.cu와 비교해 성능 차이가 거의 없었으며, 500회 반복 시 5~10ms 정도 우세한 수준에 그침. 결국 가중치 사전 로드로 memcpy를 줄이려는 접근은 큰 의미가 없다고 판단하고 my_lirpa_mempool.cu(v1-1-1)로 롤백.

- time_advantage: 5~10ms ms (500회 반복 누적 시간 기준, crown_test_preload_fix가 my_lirpa_mempool_preload_on_gpu.cu 대비 우세한 정도)
- 한계: preload 접근은 가중치 memcpy 절감 효과가 미미하여 실질적인 성능 개선으로 이어지지 않음.

## S04C05 [측정 결과/계획·요청] p14
v1-1-1-1: 메모리풀에 cuBlas 도입. 메모리풀 기반 구현에 cuBlas를 도입하여 matvec, matmul 연산을 기존 자체 계산 대신 cuBlas로 대체. 파일: my_lirpa_mempool_cuBlas.cu. 이 구간에서는 도입 사실만 기록되어 있고 측정 결과는 아직 없음.

- 한계: 이 구간에서는 도입 사실만 기록되었으며 측정 결과는 아직 없음.

## S05C01 [측정 결과/관측 기록] p15
my_lirpa_mempool_cuBlas.cu와 my_lirpa_mempool.cu(v1-1-1)의 500회 반복 실행 시간 비교. my_lirpa_mempool_cuBlas.cu(cuBlas 도입 버전)와 기존 my_lirpa_mempool.cu(v1-1-1)를 eps=1e-06, 500회 반복으로 실행한 결과, cuBlas 버전이 더 빠르게 측정되었다. 작성자는 이를 '기존 v1-1-1 대비 평균 실행당 시간이 30ms 차이난다'고 정리했다.

- Total Elapsed time (my_lirpa_mempool_cuBlas.cu, 500회): 61796,584100 ms ms (my_lirpa_mempool_cuBlas.cu, 500 iterations, eps=1e-06)
- Average time per iter (my_lirpa_mempool_cuBlas.cu, 500회): 123.593168 ms ms (my_lirpa_mempool_cuBlas.cu, 500 iterations, eps=1e-06)
- Total Elapsed time (my_lirpa_mempool.cu v1-1-1, 500회): 78755.716600 ms ms (my_lirpa_mempool.cu (v1-1-1), 500 iterations, eps=1e-06)
- Average time per iter (my_lirpa_mempool.cu v1-1-1, 500회): 157.511433 ms ms (my_lirpa_mempool.cu (v1-1-1), 500 iterations, eps=1e-06)
- 한계: p15:e121의 측정값이 명시적으로 my_lirpa_mempool_cuBlas.cu로 라벨링되어 있지 않으며, 직전 문맥(p15:e122)을 통해 추론한 것이다.

## S05C02 [측정 결과/해석] p16
메모리풀 warm up 유무에 따른 500회 반복 실행 시간 관찰. 메모리풀 warm up을 한 경우와 하지 않은 경우 모두 500회 반복 실행에서 큰 차이가 나지 않았다고 작성자가 관찰했다. 작성자는 처음 메모리 풀 할당 비용이 500회 시행에 묻혀서 차이가 크지 않았을 가능성을 추정했다.

- 한계: 정량적 수치가 아닌 정성적 관찰이며, warm up 적용/미적용의 구체적인 측정값이 이 구간에 제시되지 않았다.

## S05C03 [측정 결과/관측 기록] p15,16
my_lirpa_mempool.cu(cuBlas 없는 버전)의 500회 평균 실행 시간 변동 관찰. my_lirpa_mempool.cu(cuBlas 없는 버전)는 실행 시점에 따라 500회당 평균 120ms 대로 나올 때도 있어, 측정값이 일정하지 않고 변동이 있음을 관찰했다.

- 500회당 평균 실행 시간 (잘 나올 때): 120ms 대 ms (범위) (my_lirpa_mempool.cu (cuBlas 없는 버전), 500회 반복, 특정 실행 시점)
- 한계: 120ms 대의 구체적인 측정값(총 elapsed time, 평균 시간)은 이 구간에 기록되어 있지 않다.

## S05C04 [측정 결과/관측 기록] p16,17
hybrid_v2(v4) 500회 반복 실행과 CPU ONLY 모드 실행 시간 비교. hybrid_v2(v4)를 500회 반복 실행한 결과, CPU ONLY 모드보다 실행 속도가 느려졌다. CPU ONLY는 평균 반복당 78.296988ms, HYBRID V2는 98.832924ms로 측정되었다.

- Total Elapsed time (CPU ONLY, 500회): 39148,493900 ms ms (CPU ONLY (no GPU), 500 iterations, eps=1e-06)
- Average time per iter (CPU ONLY, 500회): 78.296988 ms ms (CPU ONLY (no GPU), 500 iterations, eps=1e-06)
- Total Elapsed time (HYBRID V2, 500회): 49416,462000 ms ms (HYBRID V2, 500 iterations, eps=1e-06)
- Average time per iter (HYBRID V2, 500회): 98.832924 ms ms (HYBRID V2, 500 iterations, eps=1e-06)
- 한계: CPU ONLY와 HYBRID V2의 측정 시점·환경이 완전히 동일한지 여부는 이 구간만으로 확인할 수 없다.

## S05C05 [문제·목표/추정·가설] p17
hybrid_v2가 CPU보다 느린 원인에 대한 추정 및 후속 실험 계획. hybrid_v2(v4)가 CPU ONLY보다 느려진 원인에 대해 작성자는 CPU 캐시 때문일 수 있다는 말을 언급했으며, 후속 실험으로 1번만 돌리는 방식으로 테스트할 계획을 세웠다.

- 한계: 원인은 아직 확인되지 않은 추정이며, 후속 실험은 계획 단계이다.

## S05C06 [측정 결과/해석] p15,17,18
cuBlas의 실행 시간 영향 평가. 작성자는 여러 측정을 거친 후 cuBlas가 실행 시간에 그렇게 큰 영향을 미치지 못하는 것 같다고 평가했다.

- 한계: 정성적 평가이며, cuBlas의 영향이 '크지 않다'고 판단한 구체적 기준이나 임계값은 제시되지 않았다.

## S05C07 [측정 결과/해석] p17
ensure_pool() 사전 메모리 할당과 cuBlas+warm up의 효과 비교. my_lirpa_mempool(v1-1-1)에서 ensure_pool()을 활용해 미리 메모리 할당을 하는 것은 실행 시간에 큰 영향을 주지만, cuBlas를 쓰고 warm up하는 것과 큰 차이가 나지 않는다고 작성자가 평가했다.

- 한계: 정성적 비교이며, 각 방식의 정량적 효과 크기가 이 구간에 직접 제시되지는 않았다(관련 수치는 p18:e135, p18:e136 참조).

## S05C08 [측정 결과/관측 기록] p18
1회 실행 기준 mempool+warmup과 mempool+cuBlas+warmup의 elapsed time 비교. 1회 시도(단일 실행) 기준으로 mempool+warmup은 152,902900ms, mempool+cuBlas+warmup은 174,978400ms로 측정되어, 1회 실행에서는 cuBlas를 넣은 쪽이 더 느렸다. 반면 warm up 자체는 각 버전에서 50ms 정도 단축 효과가 있다고 작성자가 평가했다.

- Elapsed time (mempool + warmup, 1회): 152,902900 ms ms (mempool + warmup, 1회 시도)
- Elapsed time (mempool + cuBlas + warmup, 1회): 174,978400 ms ms (mempool + cuBlas + warmup, 1회 시도)
- 한계: warm up의 50ms 단축 효과는 작성자의 정성적 평가이며, 이 구간에 warm up 미적용 상태의 1회 측정값이 함께 제시되지는 않았다.

## S05C09 [문제·목표/계획·요청] p15
메모리풀 warm up 미적용 상태에서의 후속 측정 계획. 작성자는 당시 메모리풀 warm up을 하지 않은 상태였음을 인지하고, warm up을 적용한 뒤에 측정을 진행하자는 계획을 세웠다.

- 한계: 계획 단계로, 실제 수행 결과는 이 구간에 포함되지 않았다.

## S06C01 [측정 결과/해석] p19
메모리 풀 + warm up 조합 적용 후 CROWN Verification 실행 시간 측정. my_lirpa_mempool_warmup.cu에서 메모리 풀 + warm up 조합을 적용한 후 CROWN Verification 실행 시간을 측정. 작성자는 이 두 개 조합만으로 거의 200ms 단축되어 매우 효과적이라고 평가했으며, 50ms만 더 단축하면 CPU를 따라잡을 수 있다고 판단.

- Elapsed time: 128,894800 ms (mempool + warmup 적용 후 CROWN Verification 실행, Output Dimension 16)
- 한계: CPU 대비 50ms 가량 차이로 아직 CPU보다 느린 상태

## S06C02 [측정 결과/해석] p19
make_zero_matrix/make_zero_vector CPU 회귀 및 GPU 재시도 비교. make_zero_matrix, make_zero_vector를 GPU에서 CPU 실행으로 회귀하고 파일명을 optimizing으로 변경. 이후 혹시 몰라 영벡터 만들기를 다시 GPU로 해봤으나 3~5ms 시간이 늘어나 CPU로 두는 것이 좋다고 판단.

- 시간 단축량: 22 ~ 25ms 정도 줄었다 (make_zero_matrix, make_zero_vector를 CPU로 회귀한 후)
- 시간 증가량: 3~5 ms 시간이 늘어남 (영벡터 만들기를 다시 GPU로 실행했을 때)
- 한계: GPU 제로 벡터 생성이 CPU보다 느리다는 관찰은 특정 구현·환경에 국한될 수 있음

## S06C03 [측정 결과/해석] p20
결과 저장용 행렬/벡터의 불필요한 memcpy 제거. 결과가 저장될 행렬, 벡터까지 memcpy 하던 것을 제거. 불필요한 memcpy 제거로 약 10ms 정도 시간이 줄었음.

- 시간 단축량: 10ms 정도 또 줄었음 (결과 저장용 행렬/벡터의 불필요한 memcpy 제거 후)
- Elapsed time: 115,469900 ms (memcpy 제거 후 CROWN Verification 실행, Output Dimension 16)
- 한계: 10ms 단축은 근사적 표현('정도')으로 정확한 측정 오차 범위는 명시되지 않음

## S06C04 [측정 결과/해석] p20,21
가중치 및 positive_part/negative_part 가중치 GPU 메모리 사전 적재. 가중치, positive_part/negative_part를 거친 가중치를 GPU 메모리에 미리 올려놓음. 이로 인해 matmul이나 matvec에서 memcpy가 절반으로 줄고, forward를 돌 때마다 양수/음수부 가중치를 계산할 필요가 없어짐.

- Elapsed time: 103,741500 ms (가중치 GPU 사전 적재 후 CROWN Verification 실행, Output Dimension 16)
- 한계: CPU와의 유사성은 '거의'라는 정성적 표현으로, 정확한 CPU 실행 시간과의 비교 수치는 이 구간에서 제시되지 않음

## S06C05 [측정 결과/해석] p22
forward 함수 내 DTOH memcpy 극단적 제거. forward 함수에서 DTOH하는 memcpy를 극단적으로 없앴음. 그 결과 실행 시간이 64ms로 크게 감소.

- Elapsed time: 64.716000 ms (DTOH memcpy 제거 후 CROWN Verification 실행, Output Dimension 16)
- 한계: DTOH memcpy 제거의 구체적 구현 방식이나 제거 전후 memcpy 양 비교는 이 구간에서 정량적으로 제시되지 않음

## S07C01 [측정 결과/관측 기록] p24
CPU iteration 버전 CROWN 검증 테스트. 교수에게 받은 lirpa_backward_only_iteration_array.cpp를 기존 CROWN과 결과가 동일한지 테스트. 스택 메모리가 기존 128MB로 부족하여 2GB로 늘려서 컴파일·실행함.

- Elapsed time: 193,665600 ms ms (CPU ONLY iteration ver, eps = 1e-06, mempool + warmup 없음)
- 한계: 스택 메모리 128MB로는 컴파일·실행 불가하여 2GB로 증설 필요했음.

## S07C02 [측정 결과/관측 기록] p24,25
GPU backward 파이프라인(mempool + warmup) 실행 결과. backward GPU 파이프라인을 완성하고 mempool + warmup 조건으로 실행. CPU iteration 버전(193ms) 대비 확실히 빠름. 다만 실시간성 확보를 위해서는 마이크로초~수 밀리초 내로 제한해야 함.

- Elapsed time: 26,529100 ms ms (GPU, mempool + warmup, eps = 1e-06, Output Dimension 16)
- 한계: 현재 26ms 수준으로, 실시간성 요구(마이크로초~수 밀리초)에는 아직 도달하지 못함.

## S07C03 [측정 결과/관측 기록] p25,26
cudaDeviceSynchronize 제거에 따른 시간 변화. 필요한 부분을 제외하고 cudaDeviceSynchronize 동기화 코드를 제거함. 제거 후 elapsed time이 약 1ms 감소함.

- Elapsed time: 25,457200 ms ms (GPU, mempool + warmup, cudaDeviceSynchronize 제거 후, eps = 1e-06)
- 한계: 감소폭이 1ms 수준으로 작음.

## S07C04 [측정 결과/관측 기록] p26
cuBlas 도입 결과 (성능 저하). cuBlas를 도입했으나 오히려 결과가 안좋아짐. cuBlas/cuSparse는 표준 라이브러리가 아니고 미리 컴파일된 거대 소스코드 덩어리라서 -lcublas, -lcusparse 컴파일 옵션 링크가 필요함.

- 한계: cuBlas 도입 효과 부정적. 원인 분석 및 대안 필요.

## S07C05 [측정 결과/관측 기록] p23
GPU VRAM 사용량 Nsight System 프로파일링. Nsight System으로 GPU VRAM 사용량을 측정한 결과 310.00 MiB가 나왔음. Gemini가 메모리 풀 기준 계산한 114MB보다 크지만, CUDA Context 생성(CUDA API 첫 호출 시) 및 런타임 오버헤드로 설명 가능. 300MB 내는 VRAM에게 큰 값이 아니라고 판단.

- GPU VRAM Usage: 310.00 MiB MiB (Nsight System 프로파일러 측정, GPU WDDM SYSMEM (0000:01:00.0 - NVIDIA GeForce))
- 한계: 코드 계산값(114MB)과 프로파일러 측정값(310MiB) 간 차이가 CUDA Context 오버헤드로 설명되나 정확한 분해는 아님.

## S07C06 [문제·목표/계획·요청] p23
CROWN 알고리즘 범위 재검토 필요성 (계획). CROWN 알고리즘의 범위를 잘못 규정했을 가능성 제기. partial order / total order 접근법 구분이 필요하며, 당장 중요한 것은 아니지만 미리 바꾸면 좋음.

- 한계: 아직 조사 전 상태. 후속 확인 필요.

## S07C07 [문제·목표/해석] p23
lirpa_backward_only 재귀 버전의 GPU 부적합성. 현재 lirpa_backward_only는 recursive한 버전으로, 재귀 실행이라 병렬 처리가 안되어 GPU에 맞지 않음. lirpa_backward_bound만 완성한 뒤 테스트할 예정.

- 한계: 아직 lirpa_backward_bound 완성 및 테스트 전.

## S08C01 [측정 결과/관측 기록] p27
CROWN Verification 기준 측정. CROWN verification 실행 시 elapsed time과 출력 차원을 측정. eps=1e-06, mempool+warmup 조건에서 수행.

- Elapsed time: 40,050500 ms ms (Starting CROWN Verification (eps = 1e-06) mempool + warmup)
- Output Dimension: 16 차원 (CROWN Verification 실행 결과)
- 한계: 단일 측정값으로, 반복 실험이나 통계적 정보는 없음.

## S08C02 [측정 결과/관측 기록] p28
cuSPARSE SpMV 적용 실험. cuSparse를 도입하여 벡터 곱(spMV)만 적용. W, W_pos, W_neg를 CSR 형식으로 변환하고 network_forward()의 dense matvec를 cuSPARSE SpMV로 교체. 기존 matmul은 유지.

- Elapsed time: 27,107400 ms ms (cuSPARSE SpMV 적용 후 CROWN Verification (eps = 1e-06) mempool + warmup, Output Dimension: 16)
- 한계: 기존 matmul이 아직 유지된 부분적 적용 상태이며, 단일 측정값임.

## S08C03 [측정 결과/관측 기록] p29
cuSPARSE spMM 확장 실험. spMM(Sparse 행렬곱)까지 도입한 후 결과를 확인. 약 27ms로 측정되어 4단계 최적화 최소 24ms 대비 느린 결과. 작성자는 계속 진행 여부와 cuSPARSE의 최적화 효과에 대해 의문을 제기함.

- Elapsed time: 27ms 정도 ms (spMM 도입 후 CROWN Verification 실행)
- 한계: 근사치('정도')로 기록되었으며, 계속 진행 여부는 결정되지 않음.

## S08C04 [측정 결과/계획·요청] p29
cuSPARSE 실험 방향에 대한 AI 조언. AI 조언. cuSPARSE를 계속 확장하지 않고 24ms baseline으로 돌아가며, cuSPARSE는 별도 실험으로 분리할 것을 제안. 느려진 이유는 cuSPARSE의 최적화 문제가 아니라 현재 workload가 cuSPARSE에 불리하기 때문이라고 분석.

- 한계: AI의 분석과 제안이며, 실제 실험으로 검증되지 않음.

## S09C01 [측정 결과/해석] p30,31
cuBLAS/cuSPARSE 실험 및 현재 모델 규모에서의 배제 결정. 현재 모델 크기와 호출 패턴에서 cuBLAS·cuSPARSE(SpMV/SpMM)를 실험했으나 dense 기준보다 느려져 기준 코드에서는 제외했다. network_forward()가 아직 GPU 파이프라인으로 최적화되지 않은 점, 모델·레이어 수가 작아 descriptor/buffer 처리 및 launch 오버헤드가 계산 이득보다 큰 점이 원인으로 분석된다. 공식 결과는 24ms dense baseline으로 유지하고 cuSPARSE는 별도 benchmark 옵션으로 비교하는 방향으로 정리했다.

- Elapsed time with cuSPARSE SpMV/SpMM: 38.580600 ms ms (cuSPARSE SpMV/SpMM 적용, Output Dimension 16, CROWN Verification (eps = 1e-06), mempool + warmup)
- 공식 유지 기준 dense baseline: 24 ms ms (dense matvec/matmul 유지 기준 코드, 공식 결과로 유지하기로 결정)
- 한계: cuBLAS·cuSPARSE의 기본적인 행렬곱 함수만 실험했으며, 행렬곱 변형 함수들은 아직 시도하지 않음. 추후 Nsight compute 분석과 함께 진행 필요.

## S09C02 [측정 결과/해석] p31
backward_bound_gpu() 커널 퓨전 1차 적용. backward_bound_gpu()에만 먼저 커널 퓨전을 진행했고, 23ms 후반대까지 나오는 결과를 확인했다. 컴퓨터 사정에 따라 26ms까지 뛰는 경우도 있지만, 중요한 것은 23ms 정도가 보인다는 점이다.

- Elapsed time after backward_bound_gpu kernel fusion: 23.925200 ms ms (backward_bound_gpu() 커널 퓨전 적용, CROWN Verification (eps = 1e-06), mempool + warmup, Output Dimension 16)
- 한계: backward_bound_gpu()에만 적용한 1차 결과이며, 이후 lirpa_forward_bound 전체 함수로 확장 필요.

## S09C03 [측정 결과/해석] p32
lirpa_forward_bound 함수군 커널 퓨전 및 추가 최적화. lirpa_forward_bound에서 사용하는 함수들을 커널 퓨전하고, backward_bound_gpu의 affine 중복 계산 삭제, 포인터 스와핑, backward 행렬곱 커널 2개 통합, build_coeff_pair_fused_gpu, backward_bias_pair_fused_gpu 커널 퓨전을 적용했다. 그 결과 22ms 초반까지 나오며, 늦으면 28ms까지도 나오지만 보통은 24ms 중반 정도이고 22ms 보이는 비율이 늘었다.

- Elapsed time after lirpa_forward_bound kernel fusion and optimizations: 22.342900 ms ms (lirpa_forward_bound 함수들 커널 퓨전 및 5개 최적화 적용, CROWN Verification (eps = 1e-06), mempool + warmup, Output Dimension 16)
- 한계: 실행 시간이 컴퓨터 사정에 따라 변동하며(22~28ms 범위), 보통 24ms 중반 정도로 관찰된다. 최저점(28ms)이 잘 안 보이게 되었다는 것은 체감상의 평가이다.

## S10C01 [측정 결과/해석] p33,34
커널 퓨전 + set_shape 내장 결과. 커널 퓨전 및 set_shape 함수를 다른 커널 함수에 직접 내장하여 CROWN 검증 실행 시간을 단축. 기존 22,157400 ms에서 21.375300 ms로 감소하며 최저 21ms대 진입 확인. 5번까지 적용하고 최고 성능이 나왔다는 작성자 노트가 있음.

- Elapsed time: 22,157400 ms ms (커널 퓨전 + set_shape 내장 전 baseline)
- Elapsed time: 21.375300 ms ms (커널 퓨전 + set_shape 내장 후)
- 한계: 5번의 구체적 의미(적용 대상·횟수 기준)는 원문에 명시되지 않음.

## S10C02 [측정 결과/해석] p35
alpha/beta 메모리 풀 적용. forward-backward 간에 전달되는 alpha, beta 값을 메모리 풀에 올려 전달 과정을 삭제하고, 레이어별 alpha/beta GPU 버퍼를 메모리 풀에 추가하여 H2D 복사를 제거, D2D 복사 4회를 1회로 통합. CROWN 검증 실행 시간 21,018800 ms 달성.

- Elapsed time: 21,018800 ms ms (alpha/beta 메모리 풀 적용 후)
- 한계: 단일 측정값으로 반복 측정 분산은 원문에 없음.

## S10C03 [측정 결과/확인 필요] p36
bias 메모리 풀 관리 및 안정성 관찰. bias도 메모리 풀에서 관리하도록 변경. 성능 최고점이 올랐는지는 불확실하나 21~23ms 내에서 (특히 21ms) 꽤 안정적으로 동작함.

- Elapsed time 범위: 21~23ms ms (bias 메모리 풀 관리 적용 후 (특히 21ms에서 안정적))
- 한계: 최고점 향상 여부는 불확실하며, 안정성 관찰은 정량적 분산 수치가 아닌 정성적 서술임.

## S10C04 [측정 결과/확인 필요] p36
memcpy 호출 횟수 감소 (109→94). NVTX 구간 내에서 memcpy 호출 횟수가 109회에서 94회로 감소함. 원문에 명시적 변경 사항은 없으며, bias 메모리 풀 관리 변경(p36:e238) 이후 관찰된 수치로 이전 변경들의 누적 효과일 가능성이 있음.

- memcpy 호출 횟수: 109 번 회 (NVTX 구간 내 변경 전)
- memcpy 호출 횟수: 94 번 회 (NVTX 구간 내 변경 후)
- 한계: 감소의 직접적 원인 변경이 원문에 명시되지 않음.

## S10C05 [측정 결과/해석] p36
forward 중간 결과 CPU memcpy 플래그 도입 및 효과. backward 호출 시 forward의 중간 결과를 CPU(메모리)에 복사(memcpy)하는 여부의 플래그를 도입. 플래그를 false로 설정 시 memcpy가 현저히 감소. cudaMemcpy 94회→54회, D2H 44회→20회, CUDA API 복사 시간 약 7.61ms→3.51ms로 감소. CROWN 검증 실행 시간 17.223600 ms 달성. 결과값은 기존과 동일하게 출력됨.

- Elapsed time: 17.223600 ms ms (forward 중간 결과 CPU memcpy 플래그 false 적용 후)
- cudaMemcpy 호출 횟수: 94회 회 (플래그 도입 전)
- cudaMemcpy 호출 횟수: 54회 회 (플래그 false 적용 후)
- D2H 호출 횟수: 44회 회 (플래그 도입 전)
- D2H 호출 횟수: 20회 회 (플래그 false 적용 후)
- CUDA API 복사 시간: 약 7.61ms ms (플래그 도입 전)
- CUDA API 복사 시간: 약 3.51ms ms (플래그 false 적용 후)
- 실행 시간 1회 측정: 약 20.22ms ms (1회 측정값)
- 한계: Elapsed time(17.223600 ms)과 실행 시간 1회 측정(약 20.22ms)이 별도로 제시되어 측정 방식 차이가 있을 수 있음.

## S11C01 [구현·방법/해석] p38,39
CUDA 커널 인자 전달 방식 - 스칼라/Enum은 memcpy 없이 전달 가능 (Gemini 설명). Gemini 설명에 따르면, CUDA 커널 발사 시 괄호 (...) 안에 들어가는 매개변수 중 ActivationType 같은 enum(내부적으로 4바이트 정수)이나 blocksPerGrid, 256 같은 스칼라 값은 cudaMalloc/cudaMemcpy 없이 CPU에서 값을 읽어 GPU의 매개변수 주머니(Constant Memory/Registers)에 값으로 전달됨. 반면 d_out, g_pool.d_weight 같은 포인터(배열/거대 행렬)는 크기가 너무 커서 cudaMemcpy로 VRAM에 미리 옮겨두고 주소만 전달해야 함. net.act[l]은 활성화 함수 타입을 나타내는 enum 값이므로 GPU 메모리에 올릴 필요 없이 커널 인자로 직접 전달 가능. 용량이 큰 덩어리는 포인터로 넘기고, 작은 숫자나 Enum은 값으로 넘길 수 있다는 것이 핵심.


## S11C02 [구현·방법/계획·요청] p38
network_forward(추론) 중간 복사 제거 적용 계획. network_forward(추론) 단계에서 중간 복사 제거를 적용할 예정. 커널 함수 호출에 net.act[l]을 사용했는데, 이는 메모리풀에 올리고 관리한 적이 없는 활성화 함수 관련 값임. 바꾸는 과정에 주목할 필요가 있음.


## S11C03 [측정 결과/관측 기록] p37,40
CROWN verification 실행 결과 (elapsed time, output dimension, 총 소요 시간 관찰). CROWN verification 실행 결과. eps = 1e-06, mempool + warmup calc 조건에서 실행. Output Dimension 16, Elapsed time 16,914800 ms 측정. 총 소요 시간은 16 초 후반으로 관찰됨. 출력은 16개 인덱스에 대한 Lower Bound, Upper Bound, inference(y) 값을 포함하며, 16개의 1 or 0이 출력(1이면 선택된 인덱스, 0이면 선택안된 인덱스)되는 형태임.

- Output Dimension: 16  (CROWN verification 실행 결과)
- Elapsed time: 16,914800 ms ms (eps = 1e-06, mempool + warmup calc)
- 총 소요 시간 (정성적 관찰): 16 초 후반  (CROWN verification 실행)

## S11C04 [측정 결과/관측 기록] p40
memcpy 호출 횟수 감소 (중간 복사 제거 적용 후 Nsight Compute 측정). 중간 복사 제거 적용 후 Nsight Compute로 측정한 결과, memcpy 호출 횟수가 66개에서 56개로 감소함. Nsight Compute에서 Call to cudaMemcpy 항목을 분석하여 memcpy 매칭 수를 비교함. 개별 memcpy 호출의 시작 시간, 종료 시간, 지속 시간도 측정됨.

- memcpy 호출 횟수 (적용 전): 66 개 (Nsight Compute 분석, Call to cudaMemcpy 항목)
- memcpy 호출 횟수 (적용 후): 56 개 (Nsight Compute 분석, Call to cudaMemcpy 항목)
- memcpy 호출 시작 시간: 0.399947s s (Nsight Compute 분석, Call to cudaMemcpy)
- memcpy 호출 종료 시간: 0.40014s s (Nsight Compute 분석, Call to cudaMemcpy)
- memcpy 호출 지속 시간: 192.590 µs µs (Nsight Compute 분석, Call to cudaMemcpy)
- memcpy 호출 1 시작 시간: 0.398832s s (Nsight Compute 분석, Call to cudaMemcpy, TID 14016)
- memcpy 호출 1 지속 시간: 26.601 μs μs (Nsight Compute 분석, Call to cudaMemcpy, TID 14016)
- memcpy 호출 2 시작 시간: 0.398862s s (Nsight Compute 분석, Call to cudaMemcpy, TID 14016)
- memcpy 호출 2 지속 시간: 1.084 ms ms (Nsight Compute 분석, Call to cudaMemcpy, TID 14016)
- memcpy 호출 3 시작 시간: 0.399947s s (Nsight Compute 분석, Call to cudaMemcpy, TID 14016)
- memcpy 호출 3 지속 시간: 192.590 μs μs (Nsight Compute 분석, Call to cudaMemcpy, TID 14016)

## S11C05 [한계/계획·요청] p37
CUDA core 점유율 50% 목표 (Nsight Compute 분석 기준). Nsight Compute 분석 결과 CUDA core 점유율이 50% 정도만 넘어도 잘된 편으로 볼 수 있다는 의견. 50% 넘기는 것을 목표로 설정.


## S11C06 [구현·방법/계획·요청] p37
외부 라이브러리 cuBlas, cuSparse 미도입 결정. 외부 라이브러리 cuBlas, cuSparse는 일단은 도입하지 않기로 결정. 나중에 할 것이 없으면 도입하는 것으로 유보. cuBlas는 진짜 다 하고 테스트해보는 것으로.


## S11C07 [구현·방법/계획·요청] p37
forward + backward 진행 결정 (교수님 의견). backward_only는 하지 않고 forward + backward로 계속 진행하면서 최적화하는 것이 더 나을 것이라는 교수님 의견. forward만 하는 것보다 backward가 잘 진행되고 있어 계속 진행하기로 함. 원래 f, b, b_only가 신경망 특성 때문에 거의 결과가 비슷하다는 점을 고려하여, 시간이 상대적으로 덜 걸리는 forward만 하는 방안도 검토되었으나 backward가 잘 진행되고 있어 forward + backward로 계속 진행.


## S12C01 [실험 조건/관측 기록] p41
backward_bound_gpu 내 affine_min/max GPU 실행으로 메모리 복사 최소화. backward_bound_gpu에서 affine_min/max까지 GPU에서 실행되도록 개선하여 GPU→CPU 메모리 복사를 최소화함. 기존에는 GPU(backward_bound_gpu 계산결과) → CPU 이후 CPU에서 affine_min/max 계산 수행 방식이었고, affine_min/max도 GPU에서 계산이 돌아가서 GPU → CPU → GPU → CPU 불필요한 복사 발생. 개선 후 Starting CROWN Verification 실행 시 elapsed time 15.828400 ms 측정됨.

- elapsed_time: 15.828400 ms ms (Starting CROWN Verification, backward_bound_gpu에서 affine_min/max까지 GPU 실행하도록 개선한 후)

## S12C02 [실험 조건/관측 기록] p41,42
affine_min/max 커널 퓨전. affine_min_gpu와 affine_max_gpu를 하나의 커널(affine_minmax_pair_gpu)로 병합하는 커널 퓨전 적용. 미리 작성해둔 퓨전 함수 사용. 커널 퓨전 완료 후 Starting CROWN Verification 실행 시 elapsed time 15.751000 ms 측정, 이전 15.828400 ms 대비 0.2ms 정도 하한선이 낮아진 것으로 관찰됨.

- elapsed_time_before_fusion: 15.828400 ms ms (Starting CROWN Verification, 커널 퓨전 적용 전 (backward_bound_gpu 내 affine_min/max GPU 실행 개선만 적용된 상태))
- elapsed_time_after_fusion: 15.751000 ms ms (Starting CROWN Verification, affine_min/max 커널 퓨전 적용 후 (affine_minmax_pair_gpu 사용))

## S12C03 [실험 조건/관측 기록] p43
Backward 메모리 복사 제거 및 초기화 최적화. Forward 단계의 입력 버퍼(x0 ± eps)를 Backward에서 재사용하여 H2D 복사 제거, default 초기 상태(단위행렬/영벡터)를 GPU 커널에서 직접 생성하여 PCIe 통신 오버헤드 최소화. 두 가지 최적화 적용 후 Starting CROWN Verification 실행 시 elapsed time 13,939000 ms 측정됨.

- elapsed_time: 13,939000 ms ms (Starting CROWN Verification, Forward 입력 버퍼 재사용 및 GPU 커널 직접 초기화 적용 후)

## S12C04 [실험 조건/확인 필요] p43
FP32 단정밀도 연산 변경 시도. backward_bound_gpu 최적화 과정에서 fp32에 단정밀도 연산으로 바꿔보기도 했음. 해당 실험의 상세 결과는 fp32에 대한 고찰 섹션에 있음.

- 한계: 이 구간에는 FP32 변경 결과 측정값이 없음. fp32에 대한 고찰 섹션 참조 필요.

## S12C05 [실험 조건/관측 기록] p44
CUDA 구현 동기. CUDA로 구현한 이유로 CUDA 프로그래밍 역량을 가진 사람이라는 점, 안테나 신경망 검증 알고리즘을 하드웨어에 구현하여 실시간 검증에 관심이 있다는 점이 언급됨. 하드웨어에 구현 가능한 실시간 검증 여지가 있다는 점은 논문의 뒷받침 논문으로 작성 가능함. CROWN 알고리즘 개발 팀에서도 GPU를 활용한 검증 가속을 한 적이 있으나 실시간 목적은 아니고 모든 경우를 따져보기 위한 병렬성 확보 관점에서 사용함.


## S12C06 [실험 조건/해석] p44
Kernel 시간 vs Compute throughput 해석에 대한 교수 질문과 재고. 교수님이 kernel 시간이 높을수록 좋다고 했는데 compute throughput도 높을수록 좋은게 아닌지 질문함. 작성자는 초기에 Nsight System에서 CPU-GPU 간 메모리 교류 비율이 줄어들면 GPU 커널 비중이 늘어나서 kernel은 높을수록 좋고, compute throughput은 쓸데없이 비효율적인 알고리즘 때문에 하드웨어 한계에 부딪혀서 그럴 수 있다고 생각했으나, 이후 Nsight System의 kernel도 단순히 높을수록 좋은게 아니라 왜 높은지까지 분석해야 한다고 재고함. CPU-GPU는 안줄어들고 비효율적인 GPU 계산 실행으로 계산 시간이 늘어난다면 그것 또한 커널 비중 증가로 이어지기 때문.


## S12C07 [실험 조건/계획·요청] p44
7번째 세미나 이후 후속 계획. 7번째 세미나(8/20) 이후 할 일로 (1) 영진이 형 진행 주제에 대해 다음주 미팅 잡고 질문, (2) 테스트 데이터 1개가 아닌 40000개 전체 데이터에 대해 테스트 진행 (forward, backward 2개에 대해, 찬희씨 컨택 필요), (3) 안테나 선택 신경망에서 sparse vs dense 행렬 성능 비교 및 CPU 순차처리 vs GPU 병렬처리 비교, (4) FP32로 바꾸었을 때 Nsight Compute 분석 (캐시 히트율, 처리량) 및 FP32 이용 시 64-32 변환 정밀도 보장 방법 있으나 실제 결과 다를 수 있어 재테스트 필요.

- 한계: 계획 단계로 아직 실행되지 않음.

## S13C01 [측정 결과/해석] p45
Nsight System kernel 비중· throughput 지표 해석 정정. 이전 발표에서 Nsight System의 kernel 비중은 클수록 좋고 Nsight Compute의 compute throughput은 높아지면 좋다고 말했으나, kernel 실행 시간이 잘못된 커널 작성으로 느려져도 kernel 비중이 증가할 수 있으므로 kernel 비중 증가 원인을 따져봐야 한다는 정정. Compute throughput과 메모리 throughput도 최적화 성과인지 부진인지 원인을 분석한 뒤 판단하는 것이 옳다고 정정함.

- 한계: 지표 해석 원칙을 정정했으나 실제 데이터에 대한 원인 분석은 후속 작업으로 남음.

## S13C02 [측정 결과/해석] p45,46
Forward/Backward 40000 루프 비교 및 D2H 복사 오버헤드 원인. crown_test_optimizing과 동일하게 lirpa_forward_bound_imp1(forward)과 lirpa_backward_bound(backward)를 40000 루프 각 eps 테스트로 반복 실행한 결과 forward가 backward보다 느렸음. 원인은 중간 레이어 결과 저장/복사 플래그(flag=true 시 중간 결과 복사)였으며, 단일 실행에서는 D2H 오버헤드가 연산 오버헤드를 넘지 않았으나 40000번 * 각 eps 반복에서 D2H 복사 오버헤드가 backward→forward 계산 이득을 넘어버림.

- 한계: 원인은 플래그로 특정했으나 플래그 변경 전/후의 정량적 비교 수치는 이 구간에서 제시되지 않음.

## S13C03 [측정 결과/해석] p46
Forward/Backward 복사 제어 플래그 분리 및 forward 최적화. forward의 플래그를 3개(중간 결과 layer_bound 복사용, final_affine 복사용, 최종 결과 복사용)로 분리하고, backward 플래그도 layer_bound와 backward result의 affine_bound를 같이 복사할지 제어하는 플래그를 각각 2개로 분리함. 또한 forward 최종 계산(result -> lower/upper bounds)을 D2H 복사 하는 데서 비효율적인 부분이 있어 정확한 테스트를 위해 최적화를 진행함. 결과적으로 backward가 forward보다 빠른 문제는 해결됨(forward 최적화해서).

- 한계: 해결은 선언되었으나 해결 후 forward/backward 정량 비교 수치는 이 구간에 없음.

## S13C04 [측정 결과/해석] p46
단일 테스트 memcpy 호출 횟수 및 복사 대상 재검토. 단일 데이터에 대한 memcpy 호출 횟수는 forward/backward 각각 6번. 초기에는 입력 데이터 ±eps, 가중치, bias, upper bound 결과, lower bound 결과 해서 총 6개로 생각했으나, 가중치와 bias는 nvtx 찍기 전에 한번에 올라가고(prepare network on gpu), 입력 데이터 ±eps와 upper/lower bound 결과 해서 총 4번만 복사해야 하는 것 아닌지 재검토. 신경망 추론에서는 입력값(H2D), 추론값(D2H) 복사가 발생하고, CROWN에서는 입력 데이터 ±eps와 upper/lower bound 해서 총 6개.

- 단일 데이터 memcpy 호출 횟수: 6번 회 (단일 데이터, forward/backward 각각)
- 한계: 4개로 추정했으나 실제 호출 횟수는 6번으로 관측되어 차이가 있음. 원인 규명은 후속 작업.

## S13C05 [측정 결과/확인 필요] p46
단일 테스트 Nsight System memcpy 및 커널 호출 프로파일. 단일 테스트 기준 Nsight System 프로파일에서 cudaMemcpy 6회( avg 182.834 μs, med 21.129 μs, min 9.434 μs, max 932.351 μs, stddev 368.454 μs), cudaLaunchKernel 38회( avg 10.788 μs, med 5.229 μs, min 3.159 μs, max 37.064 μs, stddev 9.384 μs), cudaMemset 4회( avg 16.547 μs, med 7.845 μs, min 4.648 μs, max 45.852 μs, stddev 19.644 μs), cuLibraryGetKernel 12회( avg 500 ns, med 362 ns, min 269 ns, max 1.509 μs, stddev 374 ns), cuKernelGetName 38회( avg 112 ns, med 92 ns, min 73 ns, max 251 ns, stddev 42 ns).

- cudaMemcpy Num Calls: 6 회 (단일 테스트)
- cudaMemcpy Avg: 182.834 μs μs (단일 테스트)
- cudaMemcpy Med: 21.129 μs μs (단일 테스트)
- cudaMemcpy Min: 9.434 μs μs (단일 테스트)
- cudaMemcpy Max: 932.351 μs μs (단일 테스트)
- cudaMemcpy StdDev: 368.454 μs μs (단일 테스트)
- cudaLaunchKernel Num Calls: 38 회 (단일 테스트)
- cudaLaunchKernel Avg: 10.788 μs μs (단일 테스트)
- cudaLaunchKernel Med: 5.229 μs μs (단일 테스트)
- cudaLaunchKernel Min: 3.159 μs μs (단일 테스트)
- cudaLaunchKernel Max: 37.064 μs μs (단일 테스트)
- cudaLaunchKernel StdDev: 9.384 μs μs (단일 테스트)
- cudaMemset Num Calls: 4 회 (단일 테스트)
- cudaMemset Avg: 16.547 μs μs (단일 테스트)
- cudaMemset Med: 7.845 μs μs (단일 테스트)
- cudaMemset Min: 4.648 μs μs (단일 테스트)
- cudaMemset Max: 45.852 μs μs (단일 테스트)
- cudaMemset StdDev: 19.644 μs μs (단일 테스트)
- cuLibraryGetKernel Num Calls: 12 회 (단일 테스트)
- cuLibraryGetKernel Avg: 500 ns ns (단일 테스트)
- cuLibraryGetKernel Med: 362 ns ns (단일 테스트)
- cuLibraryGetKernel Min: 269 ns ns (단일 테스트)
- cuLibraryGetKernel Max: 1.509 μs μs (단일 테스트)
- cuLibraryGetKernel StdDev: 374 ns ns (단일 테스트)
- cuKernelGetName Num Calls: 38 회 (단일 테스트)
- cuKernelGetName Avg: 112 ns ns (단일 테스트)
- cuKernelGetName Med: 92 ns ns (단일 테스트)
- cuKernelGetName Min: 73 ns ns (단일 테스트)
- cuKernelGetName Max: 251 ns ns (단일 테스트)
- cuKernelGetName StdDev: 42 ns ns (단일 테스트)
- 한계: 단일 테스트 기준 프로파일로, 반복 테스트나 전체 실행 프로파일과 직접 비교하려면 추가 분석 필요.

## S13C06 [측정 결과/해석] p47
프로그램 전체 memcpy 호출 횟수 분해 및 외부 복사 추정. 프로그램 전체 실행에서 memcpy 호출은 총 18번. CROWN 단계 6번을 빼면 외부에서 12번 발생. 외부 12번은 가중치 행(3개, 신경망 구조가 256 * 491 * 491 * 16 이니까 * 개수만큼) 관련으로, (가중치 + 가중치 양수부/음수부 + bias) * 3으로 추정.

- 프로그램 전체 memcpy 호출 횟수: 18번 회 (프로그램 전체 실행)
- CROWN 단계 memcpy 호출 횟수: 6번 회 (CROWN 단계)
- 외부 memcpy 호출 횟수: 12번 회 (전체 18번 - CROWN 6번)
- 한계: 추정에 기반하며 실제 복사 대상별 구분은 추가 검증 필요.

## S13C07 [측정 결과/확인 필요] p46,47
Forward/Backward 전체 실행 시간 비교 및 미해결 질문. 40000개 데이터에 대해 forward 전체 실행 시간 약 64s, backward 전체 실행 시간 약 98s. cudaMemcpy 호출 횟수는 forward/backward 동일하게 240000회임에도 forward의 cudaMemcpy 시간이 훨씬 많이 잡힘. 단일 테스트에서는 forward/backward 시간 차이가 3배 나는데, 92만개 데이터를 돌리면 비슷해지는 이유(약 1분 정도 차이)에 대한 질문은 아직 해결 못함.

- forward 전체 실행 시간: 약 64s s (40000개의 데이터에 대해서만 테스트)
- backward 전체 실행 시간: 약 98s s (40000개의 데이터에 대해서만 테스트)
- 92만개 데이터 forward/backward 시간 차이: 1분 정도 차이 없음 (92만개의 데이터를 돌렸을 때)
- 한계: 단일 테스트 3배 차이와 전체 실행 유사성 간 괴리 미해결. Nsight System 분석으로 접근했으나 원인 규명되지 않음.

## S13C08 [측정 결과/확인 필요] p47
Forward CROWN_SWEEP Nsight System 프로파일 (64.064s). Forward CROWN_SWEEP 전체 실행 시간 64.064s. cudaMemcpy가 79.7%(40.726s, 240000회, avg 169.691 μs, med 18.120 μs, min 3.599 μs, max 7.616 ms, stddev 316.565 μs)로 가장 큰 비중. cudaLaunchK 17.0%(8.698s, 1520000회), cudaMemset 3.0%(1.523s, 160000회), cuKernelGetN 0.4%(184.318ms, 1520000회), cuLibraryGetK 0.0%(7.814 μs, 12회).

- Total Time: 64.064 s s (CROWN_SWEEP forward)
- cudaMemcpy Total Time: 40.726 s s (CROWN_SWEEP forward)
- cudaMemcpy 비율: 79.7% % (CROWN_SWEEP forward)
- cudaMemcpy Num Calls: 240000 회 (CROWN_SWEEP forward)
- cudaMemcpy Avg: 169.691 μs μs (CROWN_SWEEP forward)
- cudaMemcpy Med: 18.120 μs μs (CROWN_SWEEP forward)
- cudaMemcpy Min: 3.599 μs μs (CROWN_SWEEP forward)
- cudaMemcpy Max: 7.616 ms ms (CROWN_SWEEP forward)
- cudaMemcpy StdDev: 316.565 μs μs (CROWN_SWEEP forward)
- cudaLaunchK Total Time: 8.698 s s (CROWN_SWEEP forward)
- cudaLaunchK 비율: 17.0% % (CROWN_SWEEP forward)
- cudaLaunchK Num Calls: 1520000 회 (CROWN_SWEEP forward)
- cudaLaunchK Avg: 5.722 μs μs (CROWN_SWEEP forward)
- cudaLaunchK Med: 3.843 μs μs (CROWN_SWEEP forward)
- cudaLaunchK Min: 2.785 μs μs (CROWN_SWEEP forward)
- cudaLaunchK Max: 6.908 ms ms (CROWN_SWEEP forward)
- cudaLaunchK StdDev: 9.200 μs μs (CROWN_SWEEP forward)
- cudaMemset Total Time: 1.523 s s (CROWN_SWEEP forward)
- cudaMemset 비율: 3.0% % (CROWN_SWEEP forward)
- cudaMemset Num Calls: 160000 회 (CROWN_SWEEP forward)
- cudaMemset Avg: 9.518 μs μs (CROWN_SWEEP forward)
- cudaMemset Med: 5.344 μs μs (CROWN_SWEEP forward)
- cudaMemset Min: 2.696 μs μs (CROWN_SWEEP forward)
- cudaMemset Max: 7.272 ms ms (CROWN_SWEEP forward)
- cudaMemset StdDev: 20.329 μs μs (CROWN_SWEEP forward)
- cuKernelGetN Total Time: 184.318 ms ms (CROWN_SWEEP forward)
- cuKernelGetN 비율: 0.4% % (CROWN_SWEEP forward)
- cuKernelGetN Num Calls: 1520000 회 (CROWN_SWEEP forward)
- cuKernelGetN Avg: 121 ns ns (CROWN_SWEEP forward)
- cuKernelGetN Med: 96 ns ns (CROWN_SWEEP forward)
- cuKernelGetN Min: 70 ns ns (CROWN_SWEEP forward)
- cuKernelGetN Max: 196.708 μs μs (CROWN_SWEEP forward)
- cuKernelGetN StdDev: 398 ns ns (CROWN_SWEEP forward)
- cuLibraryGetK Total Time: 7.814 μs μs (CROWN_SWEEP forward)
- cuLibraryGetK 비율: 0.0% % (CROWN_SWEEP forward)
- cuLibraryGetK Num Calls: 12 회 (CROWN_SWEEP forward)
- cuLibraryGetK Avg: 651 ns ns (CROWN_SWEEP forward)
- cuLibraryGetK Med: 449 ns ns (CROWN_SWEEP forward)
- cuLibraryGetK Min: 242 ns ns (CROWN_SWEEP forward)
- cuLibraryGetK Max: 1.903 μs μs (CROWN_SWEEP forward)
- cuLibraryGetK StdDev: 545 ns ns (CROWN_SWEEP forward)
- 한계: 프로파일 수집 조건(반복 횟수, eps 개수 등)이 명시되지 않음.

## S13C09 [측정 결과/확인 필요] p47
Backward CROWN_SWEEP Nsight System 프로파일 (98.995s). Backward CROWN_SWEEP 전체 실행 시간 98.995s. cudaMemcpy 11.332s(240000회, avg 47.216 μs, med 24.612 μs, min 3.642 μs, max 7.404 ms, stddev 51.755 μs), cudaLaunchh 18.527s(2160000회, avg 8.577 μs, med 7.208 μs, min 2.779 μs, max 7.565 ms, stddev 14.922 μs), cudaMemset 2.121s(160000회, avg 13.259 μs, med 8.447 μs, min 2.684 μs, max 2.041 ms, stddev 14.919 μs), cuKernelGetN 392.770ms(2160000회, avg 181 ns, med 160 ns, min 74 ns, max 222.425 μs, stddev 713 ns), cuLibraryGetK 16.718 μs(18회, avg 928 ns, med 638 ns, min 304 ns, max 2.763 μs, stddev 767 ns).

- Total Time: 98.995 s s (CROWN_SWEEP backward)
- cudaMemcpy Total Time: 11.332 s s (CROWN_SWEEP backward)
- cudaMemcpy Num Calls: 240000 회 (CROWN_SWEEP backward)
- cudaMemcpy Avg: 47.216 μs μs (CROWN_SWEEP backward)
- cudaMemcpy Med: 24.612 μs μs (CROWN_SWEEP backward)
- cudaMemcpy Min: 3.642 μs μs (CROWN_SWEEP backward)
- cudaMemcpy Max: 7.404 ms ms (CROWN_SWEEP backward)
- cudaMemcpy StdDev: 51.755 μs μs (CROWN_SWEEP backward)
- cudaLaunchh Total Time: 18.527 s s (CROWN_SWEEP backward)
- cudaLaunchh Num Calls: 2160000 회 (CROWN_SWEEP backward)
- cudaLaunchh Avg: 8.577 μs μs (CROWN_SWEEP backward)
- cudaLaunchh Med: 7.208 μs μs (CROWN_SWEEP backward)
- cudaLaunchh Min: 2.779 μs μs (CROWN_SWEEP backward)
- cudaLaunchh Max: 7.565 ms ms (CROWN_SWEEP backward)
- cudaLaunchh StdDev: 14.922 μs μs (CROWN_SWEEP backward)
- cudaMemset Total Time: 2.121 s s (CROWN_SWEEP backward)
- cudaMemset Num Calls: 160000 회 (CROWN_SWEEP backward)
- cudaMemset Avg: 13.259 μs μs (CROWN_SWEEP backward)
- cudaMemset Med: 8.447 μs μs (CROWN_SWEEP backward)
- cudaMemset Min: 2.684 μs μs (CROWN_SWEEP backward)
- cudaMemset Max: 2.041 ms ms (CROWN_SWEEP backward)
- cudaMemset StdDev: 14.919 μs μs (CROWN_SWEEP backward)
- cuKernelGetN Total Time: 392.770 ms ms (CROWN_SWEEP backward)
- cuKernelGetN Num Calls: 2160000 회 (CROWN_SWEEP backward)
- cuKernelGetN Avg: 181 ns ns (CROWN_SWEEP backward)
- cuKernelGetN Med: 160 ns ns (CROWN_SWEEP backward)
- cuKernelGetN Min: 74 ns ns (CROWN_SWEEP backward)
- cuKernelGetN Max: 222.425 μs μs (CROWN_SWEEP backward)
- cuKernelGetN StdDev: 713 ns ns (CROWN_SWEEP backward)
- cuLibraryGetK Total Time: 16.718 μs μs (CROWN_SWEEP backward)
- cuLibraryGetK Num Calls: 18 회 (CROWN_SWEEP backward)
- cuLibraryGetK Avg: 928 ns ns (CROWN_SWEEP backward)
- cuLibraryGetK Med: 638 ns ns (CROWN_SWEEP backward)
- cuLibraryGetK Min: 304 ns ns (CROWN_SWEEP backward)
- cuLibraryGetK Max: 2.763 μs μs (CROWN_SWEEP backward)
- cuLibraryGetK StdDev: 767 ns ns (CROWN_SWEEP backward)
- 한계: 프로파일 수집 조건이 명시되지 않음. forward와 backward의 cudaMemcpy 시간 차이가 호출당 평균 차이에서 비롯되는지, 호출 패턴 차이에서 비롯되는지는 추가 분석 필요.

## S14C01 [측정 결과/관측 기록] p48
CROWN sweep backward bound 기준 측정 (커맨드라인 인자 버전, 코드 변경 전). 커맨드라인 실행 인자로 forward/backward를 선택하는 방식에서 backward bound sweep의 전체 실행 시간. 코드 변경 전의 기준 측정값이다.

- total elapsed: 1560.37s 초 (커맨드라인 인자 있는 버전, backward bound, eps_list × N 전체 sweep, 코드 변경 전)
- 한계: 단일 실행 회차의 측정값이다.

## S14C02 [측정 결과/관측 기록] p48,49
커맨드라인 인자 기반 switch 방식 변경 후 backward sweep 성능 저하. 매번 forward/backward를 따로 실행하는 번거로움을 피하기 위해 커맨드라인 인자로 method를 선택하도록 코드를 바꾼 뒤, backward bound sweep 전체 시간이 증가하였다.

- total elapsed: 2164.98s 초 (커맨드라인 인자 있는 버전, backward bound, 코드 변경 후, eps_list × N 전체 sweep)
- 한계: 성능 저하의 원인은 이 시점에서 규명되지 않았다.

## S14C03 [문제·목표/해석] p48,49
성능 저하 원인 가설: bound result 구조체의 값에 의한 복사 (AI 조언). Codex(AI)의 진단: switch 스코프 바깥에 선언된 ForwardBoundResult/BbackwardBoundResult 구조체에 반복문 내부에서 CROWN 결과를 할당할 때 값에 의한 복사(value copy)가 발생하여 성능이 저하된다는 가설.

- 한계: AI 조언에 기반한 가설로, 작성자의 독립적 검증 결과는 이 카드에 포함되지 않는다.

## S14C04 [측정 결과/관측 기록] p50
Fix: switch case 내부 스코프 선언 적용 및 NVCC 컴파일 문제 해결 후 backward 결과. ForwardBoundResult/BbackwardBoundResult를 switch 바깥 선언 대신 각 case 안에서 선언과 동시에 생성하도록 변경하여 값에 의한 복사 문제를 해결하였다. NVCC 컴파일 오류가 발생하여 작은 스코프로 해결한 뒤, backward sweep 실행 시간이 기준선 수준으로 회복되었다.

- total elapsed: 1634.61s 초 (커맨드라인 인자 있는 버전, backward bound, fix 적용 후, eps_list × N 전체 sweep)
- 한계: 단일 실행 회차의 측정값이다. fix 전후의 정확한 비교는 별도 카드에서 다룬다.

## S14C05 [측정 결과/해석] p51
Fix 적용 후 forward bound sweep 실행 시간. fix 적용 후 커맨드라인 인자로 forward를 선택하여 forward bound sweep를 실행한 결과, backward 대비 낮은 전체 elapsed 시간이 기록되었다.

- total elapsed: 1342.47s 초 (커맨드라인 인자 있는 버전, forward bound, fix 적용 후, eps_list × N 전체 sweep)
- 한계: 작성자의 추정이며, 인과관계는 검증되지 않았다.

## S14C06 [측정 결과/관측 기록] p51
커맨드라인 인자 유무 및 forward/backward 간 실행 시간 비교. 커맨드라인 인자 있는 버전과 없는 버전, forward와 backward 간 실행 시간을 여러 회차에 걸쳐 비교하였다. 인자 없는 backward는 기준선(1560.37s)과 유사한 값을 보였고, 인자 없는 forward가 가장 낮았다.

- total elapsed: 1560.37s 초 (커맨드라인 인자 없는 버전, backward bound, eps_list × N 전체 sweep)
- total elapsed: 1561.72s 초 (커맨드라인 인자 없는 버전, backward bound, 다른 실행 회차, eps_list × N 전체 sweep)
- total elapsed: 1460.60s 초 (커맨드라인 인자 없는 버전, forward bound, eps_list × N 전체 sweep)
- total elapsed: 1564.18s 초 (커맨드라인 인자 있는 버전, backward bound, fix 적용 후, 다른 실행 회차, eps_list × N 전체 sweep)
- 한계: 실행 회차별로 값이 변동되므로 단일 비교로 일반화하기 어렵다.

## S14C07 [한계/확인 필요] p51
실행 회차 간 elapsed 시간 변동성 관찰. 동일 조건(커맨드라인 인자 유무, forward/backward)에서도 실행 회차에 따라 전체 elapsed 시간이 변동되는 현상을 관찰하였다. 약 100초 수준의 차이와 급격한 변동(2187초, 1636초)이 기록되었다.

- total elapsed: 1560.37s 초 (커맨드라인 인자 없는 버전, backward, 1회차)
- total elapsed: 1561.72s 초 (커맨드라인 인자 없는 버전, backward, 다른 회차)
- total elapsed: 1564.18s 초 (커맨드라인 인자 있는 버전, backward, fix 적용 후, 다른 회차)
- total elapsed: 2187초 초 (갑작스러운 실행 회차, 구체적 조건 미상)
- total elapsed: 1636초 초 (갑작스러운 실행 회차, 구체적 조건 미상)
- 한계: 변동 원인이 규명되지 않았으며, 구체적 조건이 기록되지 않은 측정값(2187초, 1636초)은 재현 확인이 어렵다.

## S15C01 [측정 결과/추정·가설] p52
실행 인자 및 switch문 적용에 따른 다중 데이터 전체 실행 시간 비교. 실행 인자를 추가한 forward와 실행 인자가 없는 backward의 전체 실행 시간을 비교 측정하였다. 실행 인자 있는 forward는 1360.22s, 실행 인자 없는 backward는 1447.17s로 측정되었다. 실행 인자 추가와 switch문 사용이 성능에 영향을 주었는지 여부가 의문으로 남아 있다.

- total elapsed: 1447.17s s (실행 인자 없는 backward, eps=9.0e-03, T=1.0e-02/1.0e-01/1.0e+00)
- total elapsed: 1360.22s s (실행 인자 있는 forward, eps=9.0e-05, 1.0e-02, T=1.0e-01/1.0e+00)
- total elapsed: 1375.97s s (OCR 추출값 (실행 조건 명시 없음))
- ⚠ 서술 수치가 원문 구간에 없음: 9.0e-05
- 한계: forward와 backward가 서로 다른 실행 조건(인자 유무, eps 값)에서 측정되어 직접 비교에 한계가 있음

## S15C02 [구현·방법/해석] p54
추론·bound 계산·검증 단계별 분리 측정 방식 도입. 기존의 elapsed 타임(전체 계산 시간) 측정 방식에서 추론(inference), bound 계산, 검증(verification) 각 단계 끝에 시간을 개별 기록하는 방식으로 변경하였다. elapsed 타임과 3개 누적 시간 합 사이에 미세한 오차가 있어, 3개 누적 시간을 더한 값을 더 올바른 시간으로 판단하여 채택하였다.

- 한계: 단계별 누적 시간 합이 elapsed 타임과 미세한 오차를 보이는 원인에 대한 분석은 없음

## S15C03 [측정 결과/계획·요청] p53,54
Forward/Backward 92만 번 실행 20회 반복 측정 및 통계 산출. 추론 + bound + 검증을 분리하여 각각 측정하도록 변경한 후, forward와 backward에 대해 각각 20회씩 측정을 완료하였다. 튀는 값이 일부 존재하며, 각 20번 실행 시간의 평균과 분산을 구하여 슬라이드에 넣을 예정이다.

- 한계: 평균·분산 값은 아직 산출되지 않음 (슬라이드 삽입 예정 상태)

## S15C04 [구현·방법/해석] p53,54,55
단일 데이터 warm-up 영향 측정 및 검증. 단일 데이터(optimizing.exe) 측정에 warm-up 코드를 추가하여 warm-up이 측정 시간에 미치는 영향을 검증하였다. 측정은 완료되었으며, 발표에서 교수님이 초기화 비용 때문일 것이라고 언급하였다.

- 한계: warm-up 적용 전후의 정량적 시간 비교 값은 기록에 없음

## S15C05 [측정 결과/확인 필요] p55
단일 데이터 추론·bound 측정 결과. 단일 데이터에 대해 추론(inference)과 bound 계산 시간을 개별 측정하였다. elapsed time 1.6966 ms, inference time 0.1417 ms, bound time 1.5547 ms, output dimension 16으로 기록되었다. inference time은 forward 계산 시간으로, bound time은 20번 92만 번 테스트(test_all_data)의 평균 bound로 언급되었다.

- elapsed time: 1.6966 ms ms (단일 데이터, forward + bound 계산 포함)
- inference time: 0.1417 ms ms (단일 데이터, 신경망 정방향 통과 (forward 계산 시간))
- bound time: 1.5547 ms ms (단일 데이터, CROWN bound 계산 (20번 92만 번 테스트 평균 bound로 언급))
- output dimension: 16  (단일 데이터 추론 결과)
- 한계: bound time이 단일 측정값인지 20회 평균인지 불명확. 메모리 할당(128MB) 이유에 대해 '왜인지는 모르겠다'고 기록됨 (p55:e400).

## S15C06 [문제·목표/추정·가설] p52
현재 해결 중인 성능 차이 의문점. 현재 세 가지 의문점이 정리되어 있다: (1) 단일 데이터 테스트(forward/backward 3배 차이)와 달리 다중 데이터 테스트에서 forward 성능 감소가 어디서 일어나는지, (2) 구조체 값에 의한 복사 제거 후 backward가 다시 2100s 정도로 튀는 원인, (3) 실행 인자 추가 + switch문 사용이 실제 성능에 영향을 주었는지 여부.

- 한계: 의문점에 대한 원인 분석이나 실험적 검증은 아직 이루어지지 않음

## S15C07 [한계/계획·요청] p53
향후 보완 및 측정 계획. 다음 주에 보완·측정할 항목으로 Compute Throughput과 Memory Throughput의 정확한 정의(수식) 확인, forward/backward 92만 번 실행의 평균·분산 산출, 추론+bound+검증 분리 측정, 신경망 검증팀과 forward/backward 검증 비율 차이 확인, TensorRT 컴파일 결과와 nvcc 컴파일 시간 비교 등이 계획되어 있다.

- 한계: 계획 단계의 항목이 많아 실제 수행 여부는 불확실함

## S16C01 [측정 결과/해석] p58
sparse.py 대비 CUDA CROWN bound 계산 성능 비교. sparse.py와 CUDA 구현 간 CROWN bound 계산 평균 1회 성능을 비교한 결과표. forward/backward 모두 CUDA가 sparse.py 대비 약 5.3배 빠르고, 추론+bound+검증 전체 기준으로도 약 4.8~4.9배 빠르다.

- forward bound 계산 성능 (배율): 5.32x  (sparse.py 6.951 ms vs CUDA 1.307 ms)
- forward bound 계산 성능 (단축률): 81.2%  (sparse.py vs CUDA 비교)
- forward 추론 + bound + 검증 (배율): 4.86x  (sparse.py vs CUDA 비교)
- forward 추론 + bound + 검증 (단축률): 79.4%  (sparse.py vs CUDA 비교)
- backward bound 계산 성능 (배율): 5.30x  (sparse.py 8.044 ms vs CUDA 1.519 ms)
- backward bound 계산 성능 (단축률): 81.1%  (sparse.py vs CUDA 비교)
- backward 추론 + bound + 검증 (배율): 4.89x  (sparse.py vs CUDA 비교)
- backward 추론 + bound + 검증 (단축률): 79.6%  (sparse.py vs CUDA 비교)
- ⚠ 서술 수치가 원문 구간에 없음: 4.8, 4.9
- 한계: 평균 1회 결과 기준이며, 측정 조건(반복 횟수, warmup 여부 등)은 표에 명시되지 않음.

## S16C02 [측정 결과/해석] p58,59
배칭 시스템 구현으로 3배 속도 향상 (25분→8분). 멀티 스트림 배칭을 적용하여 준비된 데이터셋 실험 환경에서 단일 스트림 대비 3배 빠른 처리 시간(25분→8분)을 확인함. 실시간 환경에서는 데이터셋을 꽉 채워 넣기 어렵다는 한계가 언급됨.

- 배칭 적용 전 처리 시간: 25분  (단일 스트림)
- 배칭 적용 후 처리 시간: 8분  (멀티 스트림 배칭)
- 속도 향상 배율: 3배  (단일 스트림 대비 멀티 스트림 배칭)
- 한계: 실시간 환경에서는 데이터셋을 꽉 채워 넣기 어려워 이 배칭 방식을 그대로 적용하기 어려움.

## S16C03 [측정 결과/해석] p59
cudaGraph를 이용한 커널 캡처로 약 400초 감소. cudaGraph를 통해 추론과 bound를 따로 캡처하거나 추론+bound를 한 번에 캡처하여 커널 런치 비용을 최소화한 결과, 약 400초 정도 시간이 줄어든 것으로 보임.

- 감소된 시간: 한 400초 정도  (추론이랑 bound랑 따로 캡처했을 때 기준)
- 한계: 대략적인 감소 시간('한 400초 정도')으로 정확한 측정값이 아님. 커널 자체 최적화까지 끝난 후 고려 가능한 방법.

## S16C04 [측정 결과/해석] p56,57
warmup 후 forward/backward bound 계산 시간 유사성 확인. warmup 1회 후 측정 시 forward와 backward의 bound 계산 시간이 약 0.2ms 차이로 유사하게 나타남. 초기 초기화 비용으로 인해 3배 차이가 났던 것이 맞는지 검증하기 위해 시작한 조사임.

- warmup 후 측정 예시 1 - Elapsed time: 1.9254 ms ms (warmup 한번, mempool + warmup)
- warmup 후 측정 예시 1 - Inference time: 0.1404 ms ms (warmup 한번, mempool + warmup)
- warmup 후 측정 예시 1 - Bound time: 1.7846 ms ms (warmup 한번, mempool + warmup)
- warmup 후 측정 예시 2 - Elapsed time: 1.9171 ms ms (warmup 한번)
- warmup 후 측정 예시 2 - Inference time: 0.1409 ms ms (warmup 한번)
- warmup 후 측정 예시 2 - Bound time: 1.7760 ms ms (warmup 한번)
- warmup 후 측정 예시 3 (mempool + warmup) - eps calc success lapsed time: 1,9759 ms ms (mempool + warmup)
- warmup 후 측정 예시 3 (mempool + warmup) - Inference time: 0.1867 ms ms (mempool + warmup)
- warmup 후 측정 예시 3 (mempool + warmup) - Bound time: 1,7889 ms ms (mempool + warmup)
- forward/backward bound 시간 차이: 한 0.2ms 정도  (warmup 한번 후 측정)
- 한계: backward 측정값의 정확한 수치는 이미지 내 텍스트로 추출되지 않아 비교에 제한이 있음.

## S16C05 [측정 결과/해석] p56
40000개 데이터 평균 bound 시간과 단일 데이터 측정 비교. test_all_data.bin의 40000개 데이터에 대한 평균 bound 시간(1.91838356 ms)과 단일 데이터 측정값(Elapsed 1.9171 ms)을 비교함. 단일 데이터가 평균보다 약간 달랐으며, 작성자는 빠른 데이터가 느린 데이터보다 많았을 가능성을 추정함.

- 40000개 데이터 평균 bound 시간: 1.91838356 ms (test_all_data.bin 전체 40000개 평균)
- 40000개 데이터 개수: 40000  (test_all_data.bin)
- 40000개 데이터 합계: 76735.3424  (test_all_data.bin 전체 합계)
- 단일 데이터 측정 - Elapsed time: 1.9171 ms ms (test_all_data.bin에서 단일 데이터 추출)
- 단일 데이터 측정 - Inference time: 0.1409 ms ms (test_all_data.bin에서 단일 데이터 추출)
- 단일 데이터 측정 - Bound time: 1.7760 ms ms (test_all_data.bin에서 단일 데이터 추출)
- 평균 bound 시간(텍스트 표기): 1.91 ms 정도  (test_all_data.bin 평균)
- 한계: 단일 데이터가 평균보다 느렸는지 빨랐는지 명확한 결론은 내려지지 않음. 작성자도 '너무 깊게 파고들지 말자'고 판단.

## S16C06 [구현·방법/계획·요청] p57
GPU 벤치마크 모범 사례에 따른 측정 기준 재정립. GPU 벤치마크 모범 사례를 따라 측정 기준을 다시 세우기로 함. 실제 사이클 시간만 측정(가중치 로드 memcpy 제외), warmup 1회, CV를 통한 스파이크/분산 규명, 추론·bound 계산·검증 3가지 값을 나눠서 측정하는方針.

- 한계: 계획 단계의 내용으로 실제 적용 결과는 별도 확인 필요.

## S16C07 [측정 결과/해석] p57
warmup 적용 후 측정 시간 증가 및 스파이크 문제. NVIDIA 가이드라인에 따라 warmup을 적용했으나, 예상과 달리 측정 시간 및 각 평균 시간이 더 늘어남. forward가 warmup 이전 backward 급 속도가 나올 정도였으며, warmup을 해도 스파이크 데이터가 뜨고 10번 측정 시 평균 시간이 느려지는 문제가 발생함.

- warmup 후 스파이크 발생: warmup을 해도 스파이크 데이터가 뜨고, 가끔 10번 측정 시 평균 시간이 느려지는 문제  (warmup 적용 후 10번 측정)
- 한계: warmup이 오히려 측정 시간을 증가시키고 스파이크 문제를 해결하지 못함. 변동 계수 개선이 '느낌' 수준으로 정량화되지 않음.

## S16C08 [측정 결과/해석] p58
5.3배 차이 확인 및 스파이크·분산 문제 잔존. 5.3배 정도 차이나는 것을 확인했으나, 스파이크 문제와 평균 시간 분포가 살짝 올라간 문제가 남아있음. 클록 고정 후 재측정하거나 GPU 사용량을 측정해보는 방안이 제시됨.

- 한계: 스파이크 문제와 평균 시간 분포 상승 원인은 규명되지 않음. CV가 간헐적으로 튀는 현상으로 설명되었으나 정량적 분석은 부족.

## S16C09 [구현·방법/계획·요청] p58,59
수행한 최적화 목록 및 추가 최적화 계획. 현재까지 수행한 최적화와 추가 고려 사항을 정리. 수행: 제로 memcpy, 커널 실행 횟수 감소(커널 퓨전, 커널 graph), GPU 아키텍처 SM 구조 고려(FP64→FP32, double→float). 계획: cudaGraph를 이용한 커널 캡처, 배칭(멀티 스트림).

- 한계: cudaGraph는 커널 자체 최적화까지 끝냈을 때 고려 가능한 방법이며, 배칭은 실시간 환경에 적합하지 않을 수 있음.

## S16C10 [측정 결과/해석] p58
상세 측정 테이블 (forward/backward 40000개 데이터). forward/backward에 대해 40000개 데이터, 23개 단위로 측정한 상세 타이밍 테이블. 열 의미는 OCR상 명확하지 않으나 원문 표기 그대로 기록.

- backward 행 - 열2: 40000  (40000개 데이터, 23개 단위 측정)
- backward 행 - 열3: 23  (40000개 데이터, 23개 단위 측정)
- backward 행 - 열4: 481.7157  (40000개 데이터, 23개 단위 측정)
- backward 행 - 열5: 1.3569  (40000개 데이터, 23개 단위 측정)
- backward 행 - 열6: 479.6443  (40000개 데이터, 23개 단위 측정)
- backward 행 - 열7: 0.6957  (40000개 데이터, 23개 단위 측정)
- backward 행 - 열8: 0.0015  (40000개 데이터, 23개 단위 측정)
- backward 행 - 열9: 0.5214  (40000개 데이터, 23개 단위 측정)
- backward 행 - 열10: 0.0008  (40000개 데이터, 23개 단위 측정)
- forward 행 - 열2: 40000  (40000개 데이터, 23개 단위 측정)
- forward 행 - 열3: 23  (40000개 데이터, 23개 단위 측정)
- forward 행 - 열4: 438.3345  (40000개 데이터, 23개 단위 측정)
- forward 행 - 열5: 1.599  (40000개 데이터, 23개 단위 측정)
- forward 행 - 열6: 435.8792  (40000개 데이터, 23개 단위 측정)
- forward 행 - 열7: 0.6724  (40000개 데이터, 23개 단위 측정)
- forward 행 - 열8: 0.0017  (40000개 데이터, 23개 단위 측정)
- forward 행 - 열9: 0.4738  (40000개 데이터, 23개 단위 측정)
- forward 행 - 열10: 0.0007  (40000개 데이터, 23개 단위 측정)
- 한계: 테이블 열 헤더가 OCR에서 추출되지 않아 각 열의 의미(총 elapsed time, 평균, 분산, 표준편차와 bound 관련 세부 시간 등)를 확정할 수 없음.

## S16C11 [구현·방법/해석] p59
pinned Memory + cudaMemcpyAsync를 통한 메모리 복사 속도 최적화. 데이터셋을 통째로 넘겨주는 배칭은 실시간 환경이 아니라고 판단하여, 대신 데이터셋이 있는 부분을 pinned Memory로 지정하고 메모리 복사 속도를 최대로 늘림. cudaMemcpyAsync를 사용하기도 함.

- 한계: 정량적인 속도 향상 수치는 언급되지 않음. 실시간 환경 suitability에 대한 판단만 제시됨.

## S17C01 [측정 결과/해석] p60,61
memcpy 크기 축소 변경 및 성능 영향. 커널 memcpy에서 기존 MAXDIM(512 float) 고정 복사 대신 필요한 데이터 크기만 복사하도록 변경했으나, 전체 시간이 72초 증가하여 더 느려졌다. CPU 계산 오버헤드가 원인으로 의심된다.

- forward_cuda_graph 시간: 983.5557  (N=40000, 반복 23회, forward_cuda_graph (graph만 적용한 버전))
- 실시간 멀티스트림 forward_cuda_graph 시간: 1055.282  (N=40000, 반복 23회, 실시간 멀티스트림 forward_cuda_graph (cpu에서 필요한 만큼만 계산 후 복사))
- forward_cuda_graph_fused 시간: 1006.863  (N=40000, 반복 23회, forward_cuda_graph_fused)
- backward_cuda_graph 시간: 1134.724  (N=40000, 반복 23회, backward_cuda_graph)
- backward_cuda_graph_fused 시간: 1116.847  (N=40000, 반복 23회, backward_cuda_graph_fused)
- 실시간 멀티스트림 전체 elapsed: 1055.28s  (실시간 멀티스트림 전체 elapsed)
- infer phase 시간: 131.799s  (실시간 멀티스트림 infer phase (전체 eps × 전체 data))
- bound phase 시간: 920.844s  (실시간 멀티스트림 bound phase (전체 eps × 전체 data))
- certify phase 시간: 1,597s  (실시간 멀티스트림 certify phase (전체 eps × 전체 data))
- 단일 샘플 최소 지연시간: 1092.30 us  (B=1, 100000 샘플, Real-Time Single-Sample Latency Analysis)
- 단일 샘플 평균 지연시간: 1145.24 us  (B=1, 100000 샘플, Real-Time Single-Sample Latency Analysis)
- 단일 샘플 중앙값 지연시간 (P50): 1136.80 us  (B=1, 100000 샘플, Real-Time Single-Sample Latency Analysis)
- 단일 샘플 P95 꼬리 지연시간: 1179,00 us  (B=1, 100000 샘플, Real-Time Single-Sample Latency Analysis)
- 단일 샘플 P99 꼬리 지연시간: 1311.90 us  (B=1, 100000 샘플, Real-Time Single-Sample Latency Analysis)
- 단일 샘플 최대 지연시간: 7830.90 us  (B=1, 100000 샘플, Real-Time Single-Sample Latency Analysis)
- 단일 샘플 지터 (표준편차): 60.33 us  (B=1, 100000 샘플, Real-Time Single-Sample Latency Analysis)
- 단일 샘플 처리량: 873.18 samples/sec  (B=1, 100000 샘플, Real-Time Single-Sample Latency Analysis)
- 순수 GPU 실행 시간 (마지막 샘플): 953.70 us  (B=1, 마지막 샘플, Pure GPU Execution)
- 한계: 평가 샘플이 10만개뿐인 이유는 표본 측정 목적 (p61:e469)

## S17C02 [구현·방법/확인 필요] p60
데이터셋 pinned memory 적용. 데이터셋 호스트 메모리를 cudaHostRegister로 pinned memory로 등록하여 GPU가 CPU 메모리에 직접 접근할 수 있도록 함. 실시간 입력 속도 재현을 위한 목적.


## S17C03 [문제·목표/해석] p61
성능 저하 원인 분석. memcpy 변경 후 시간이 늘어난 원인으로 CPU 계산 오버헤드, cudaEvent 측정 오버헤드, GPU 클럭 변동 세 가지를 의심. cudaEvent는 시스템 인자(measure_gpu)로 제어 중.


## S17C04 [한계/해석] p61
후속 계획 및 제거 후보 정리. 다음 주 발표 준비 필요. 커밋 이전 그래프만 적용한 버전이 더 빠름을 확인. topk 속도는 빨라져 제외 가능성 있음. 제거할 부분으로 데이터셋 pinnedMemory, 검증(heap->stack) 배열, cudaEvent는 Nsight Compute 분석을 위해 필요함을 정리. 결국 CPU를 통해 필요한 만큼만 계산하는 부분을 지워볼 만하다고 판단.


## S18C01 [구현·방법/계획·요청] p62
측정 방식: 평균 사용. 측정 시 개별 값이 아닌 평균을 사용하여 성능을 평가하기로 함.


## S18C02 [실험 조건/계획·요청] p62
비교 대상 버전 구성: cudaGraph-only vs 최적화 적용 버전. cudaGraph만 적용된 버전과, pinned Memory 추가 + cudaEvent 제어 + 필요한 만큼 계산해서 D2H memcpy + certify topk 자료구조 스택화가 적용된 버전을 비교 대상으로 함.


## S18C03 [측정 결과/해석] p62
Graph 단일 적용 vs 최적화 버전: Forward 비교 관찰. Forward에서 graph 단일 적용이 전체적으로(추론, bound, 검증) 우수하며, 특히 infer_s에서 약 20초 차이가 난다고 관찰함.

- 한계: graph-only 버전의 측정값이 본 구간의 표에 포함되지 않아 20초 차이의 직접적 근거는 이 표에 없음.

## S18C04 [측정 결과/해석] p62
Graph 단일 적용 vs 최적화 버전: Backward 비교 관찰. Backward에서는 total 시간이 유사하며, infer에서 절약한 20초를 bound에서 graph보다 20초 빨리 수행하여 상쇄된다고 관찰함.

- 한계: graph-only 버전의 측정값이 본 구간의 표에 포함되지 않아 20초 차이의 직접적 근거는 이 표에 없음.

## S18C05 [측정 결과/계획·요청] p62
현재 방식 유지 결정 및 향후 최적화 아이디어. 현재 방식을 일단 유지하기로 결정함. 향후 infer는 벡터라 크지 않으므로 크기 계산 주석처리 고려, certify는 heap으로 다시 바꿔야 할지 검토 필요.

- 한계: 아이디어 단계로 실제 적용 결과는 없음.

## S18C06 [측정 결과/계획·요청] p62
forward_cuc 성능 측정 (N=40000, eps_count=23, 평균) - 표1. forward_cuc 방법의 성능 측정 결과 평균값. N=40000, eps_count=23 조건에서 표의 평균 행 기준.

- total_s: 1002,77848 초 (method=forward_cuc, N=40000, eps_count=23, 표의 평균 행 기준)
- infer_s: 111,18044 초 (method=forward_cuc, N=40000, eps_count=23, 표의 평균 행 기준)
- bound_s: 890,80258 초 (method=forward_cuc, N=40000, eps_count=23, 표의 평균 행 기준)
- certify_s: 0,72632 초 (method=forward_cuc, N=40000, eps_count=23, 표의 평균 행 기준)
- avg_infer_ms: 0,12085 밀리초 (method=forward_cuc, N=40000, eps_count=23, 표의 평균 행 기준)
- avg_bound_ms: 0,96827 밀리초 (method=forward_cuc, N=40000, eps_count=23, 표의 평균 행 기준)
- avg_certify_ms: 0,0008 밀리초 (method=forward_cuc, N=40000, eps_count=23, 표의 평균 행 기준)
- 한계: 개별 측정값의 편차가 존재하며, 그래프 전용 버전과의 직접 비교 수치는 이 표에 없음. 측정 횟수(행 개수)는 원문에 명시되지 않음.

## S18C07 [측정 결과/계획·요청] p62
backward_c 성능 측정 (N=40000, eps_count=23, 평균) - 표1. backward_c 방법의 성능 측정 결과 평균값. N=40000, eps_count=23 조건에서 표의 평균 행 기준.

- total_s: 1139,60662 초 (method=backward_c, N=40000, eps_count=23, 표의 평균 행 기준)
- infer_s: 115,39256 초 (method=backward_c, N=40000, eps_count=23, 표의 평균 행 기준)
- bound_s: 1023,05768 초 (method=backward_c, N=40000, eps_count=23, 표의 평균 행 기준)
- certify_s: 1,02762 초 (method=backward_c, N=40000, eps_count=23, 표의 평균 행 기준)
- avg_infer_ms: 0,12542 밀리초 (method=backward_c, N=40000, eps_count=23, 표의 평균 행 기준)
- avg_bound_ms: 1,11202 밀리초 (method=backward_c, N=40000, eps_count=23, 표의 평균 행 기준)
- avg_certify_ms: 0,00111 밀리초 (method=backward_c, N=40000, eps_count=23, 표의 평균 행 기준)
- 한계: 개별 측정값의 편차가 존재하며, 그래프 전용 버전과의 직접 비교 수치는 이 표에 없음. 측정 횟수(행 개수)는 원문에 명시되지 않음.

## S18C08 [측정 결과/계획·요청] p62
forward_cuc 성능 측정 (measure=1, 평균) - 표2. forward_cuc 방법의 성능 측정 결과 평균값. N=40000, eps_count=23, measure=1 조건에서 표의 평균 행 기준.

- total_s: 1029,827 초 (method=forward_cuc, N=40000, eps_count=23, measure=1, 표의 평균 행 기준)
- infer_s: 132,985 초 (method=forward_cuc, N=40000, eps_count=23, measure=1, 표의 평균 행 기준)
- bound_s: 894,522 초 (method=forward_cuc, N=40000, eps_count=23, measure=1, 표의 평균 행 기준)
- certify_s: 1,94402 초 (method=forward_cuc, N=40000, eps_count=23, measure=1, 표의 평균 행 기준)
- avg_infer_: 0,14455 밀리초 (method=forward_cuc, N=40000, eps_count=23, measure=1, 표의 평균 행 기준)
- avg_boun: 0,97232 밀리초 (method=forward_cuc, N=40000, eps_count=23, measure=1, 표의 평균 행 기준)
- avg_certify_r: 0,0021 밀리초 (method=forward_cuc, N=40000, eps_count=23, measure=1, 표의 평균 행 기준)
- 한계: 표1과 표2의 측정 조건 차이(measure 컬럼 유무 등)가 명시적으로 기술되지 않음. 열 이름이 일부 잘려 있음(avg_infer_, avg_boun, avg_certify_r). 측정 횟수(행 개수)는 원문에 명시되지 않음.

## S18C09 [측정 결과/계획·요청] p62
backward_c 성능 측정 (measure=1, 평균) - 표2. backward_c 방법의 성능 측정 결과 평균값. N=40000, eps_count=23, measure=1 조건에서 표의 평균 행 기준.

- total_s: 1138,49 초 (method=backward_c, N=40000, eps_count=23, measure=1, 표의 평균 행 기준)
- infer_s: 132,962 초 (method=backward_c, N=40000, eps_count=23, measure=1, 표의 평균 행 기준)
- bound_s: 1003,18 초 (method=backward_c, N=40000, eps_count=23, measure=1, 표의 평균 행 기준)
- certify_s: 1,96173 초 (method=backward_c, N=40000, eps_count=23, measure=1, 표의 평균 행 기준)
- avg_infer_: 0,14451 밀리초 (method=backward_c, N=40000, eps_count=23, measure=1, 표의 평균 행 기준)
- avg_boun: 1,0904 밀리초 (method=backward_c, N=40000, eps_count=23, measure=1, 표의 평균 행 기준)
- avg_certify_r: 0,00212 밀리초 (method=backward_c, N=40000, eps_count=23, measure=1, 표의 평균 행 기준)
- 한계: 표1과 표2의 측정 조건 차이가 명시적으로 기술되지 않음. 열 이름이 일부 잘려 있음. 측정 횟수(행 개수)는 원문에 명시되지 않음.

## S19C01 [측정 결과/계획·요청] p64
auto_LiRPA forward+backward 검증 결과. auto_LiRPA를 forward+backward 모드로 실행하여 eps별 검증 결과(T/F/ratio/time) 및 전체 소요 시간을 측정. eps 증가 시 ratio는 감소함.

- sum_of_eps_times: 107.30s s (forward+backward 모드 전체 EPS 시간 합산)
- overall_elapsed: 107.32s s (forward+backward 모드 전체 프로그램 소요 시간)
- time: 5.04s s (eps=1e-06, T=39969, F=31, ratio=0.999)
- time: 4.63s s (eps=1e-05, T=39709, F=291, ratio=0.993)
- time: 4.65s s (eps=0.0001, T=37026, F=2974, ratio=0.926)
- time: 4.64s s (eps=0.0002, T=34353, F=5647, ratio=0.859)
- time: 4.63s s (eps=0.0003, T=31826, F=8174, ratio=0.796)
- time: 4.62s s (eps=0.0004, T=29375, F=10625, ratio=0.734)
- time: 4.65s s (eps=0.0005, T=27170, F=12830, ratio=0.679)
- time: 4.71s s (eps=0.0006, T=25103, F=14897, ratio=0.628)
- time: 4.72s s (eps=0.0007, T=23179, F=16821, ratio=0.579)
- time: 4.77s s (eps=0.0008, T=21323, F=18677, ratio=0.533)
- time: 4.68s s (eps=0.0009, T=19656, F=20344, ratio=0.491)
- time: 4.62s s (eps=0.001, T=18128, F=21872, ratio=0.453)
- time: 4.64s s (eps=0.002, T=7461, F=32539, ratio=0.187)
- time: 4.61s s (eps=0.003, T=2754, F=37246, ratio=0.069)
- time: 4.63s s (eps=0.004, T=945, F=39055, ratio=0.024)
- time: 4.61s s (eps=0.005, T=286, F=39714, ratio=0.007)
- time: 4.62s s (eps=0.006, T=71, F=39929, ratio=0.002)
- time: 4.60s s (eps=0.007, T=24, F=39976, ratio=0.001)
- time: 4.57s s (eps=0.008, T=3, F=39997, ratio=0.000)
- time: 4.66s s (eps=0.009, T=1, F=39999, ratio=0.000)
- time: 4.66s s (eps=0.01, T=0, F=40000, ratio=0.000)
- time: 4.67s s (eps=0.1, T=0, F=40000, ratio=0.000)
- time: 4.67s s (eps=1.0, T=0, F=40000, ratio=0.000)
- 한계: auto_LiRPA 라이브러리 기준 결과이며, 자체 구현 커널과의 직접 비교는 아님.

## S19C02 [측정 결과/계획·요청] p64,65
auto_LiRPA forward-only 검증 결과. auto_LiRPA를 forward-only 모드로 실행하여 eps별 검증 결과(T/F/ratio/time) 및 전체 소요 시간을 측정. forward+backward 대비 전체 소요 시간이 더 길게 나타남(114.78초 vs 107.32초).

- sum_of_eps_times: 114.77s s (forward-only 모드 전체 EPS 시간 합산)
- overall_elapsed: 114.78s s (forward-only 모드 전체 프로그램 소요 시간)
- time: 5.32s s (eps=1e-06, T=39969, F=31, ratio=0.999)
- time: 5.03s s (eps=1e-05, T=39709, F=291, ratio=0.993)
- time: 4.95s s (eps=0.0001, T=37026, F=2974, ratio=0.926)
- time: 4.98s s (eps=0.0002, T=34349, F=5651, ratio=0.859)
- time: 5.00s s (eps=0.0003, T=31813, F=8187, ratio=0.795)
- time: 5.02s s (eps=0.0004, T=29366, F=10634, ratio=0.734)
- time: 5.09s s (eps=0.0005, T=27153, F=12847, ratio=0.679)
- time: 5.10s s (eps=0.0006, T=25086, F=14914, ratio=0.627)
- time: 4.99s s (eps=0.0007, T=23154, F=16846, ratio=0.579)
- time: 4.97s s (eps=0.0008, T=21288, F=18712, ratio=0.532)
- time: 4.97s s (eps=0.0009, T=19623, F=20377, ratio=0.491)
- time: 5.00s s (eps=0.001, T=18073, F=21927, ratio=0.452)
- time: 4.92s s (eps=0.002, T=7380, F=32620, ratio=0.184)
- time: 4.93s s (eps=0.003, T=2695, F=37305, ratio=0.067)
- time: 4.90s s (eps=0.004, T=905, F=39095, ratio=0.023)
- time: 4.94s s (eps=0.005, T=269, F=39731, ratio=0.007)
- time: 4.97s s (eps=0.006, T=67, F=39933, ratio=0.002)
- time: 4.91s s (eps=0.007, T=20, F=39980, ratio=0.001)
- time: 4.92s s (eps=0.008, T=2, F=39998, ratio=0.000)
- time: 4.94s s (eps=0.009, T=0, F=40000, ratio=0.000)
- time: 4.94s s (eps=0.01, T=0, F=40000, ratio=0.000)
- time: 4.97s s (eps=0.1, T=0, F=40000, ratio=0.000)
- time: 4.99s s (eps=1.0, T=0, F=40000, ratio=0.000)
- 한계: auto_LiRPA 라이브러리 기준 결과이며, 자체 구현 커널과의 직접 비교는 아님.

## S19C03 [측정 결과/계획·요청] p65
auto_LiRPA backward-only 검증 결과 (부분). auto_LiRPA를 backward-only 모드로 실행하여 eps별 검증 결과 측정. 문서에 기록된 것은 eps=1e-06 ~ 0.0002 까지 일부 행뿐이며, time은 1.09~2.67초로 다른 모드 대비 현저히 짧음.

- time: 1.09s s (eps=1e-06, T=39969, F=31, ratio=0.999)
- time: 1.56s s (eps=1e-05, T=39709, F=291, ratio=0.993)
- time: 2.67s s (eps=0.0001, T=37026, F=2974, ratio=0.926)
- time: 2.54s s (eps=0.0002, T=34353, F=5647, ratio=0.859)
- 한계: 전체 eps 범위 중 일부 행만 기록되어 있어 전체 경향 파악에 한계가 있음.

## S19C04 [측정 결과/계획·요청] p63
10주차 세미나 이후 액션 아이템 및 현황. 10주차 세미나 이후 정리: backward_only CUDA 커널 구현 완료, 최적화 적용 여부 탐색중. cudaGraph 원리 학습 필요. 찬희씨와 auto_lirpa 모델 소통 완료(repo 수령, 데이터셋·모델 동일 확인). 컴파일러 팀에 모델 공유 완료. 배칭+다중 스트림 환경에서 GPU 자원 여유 확인, 단일 스트림·단일 데이터 복사가 현재 주요 병목.

- 한계: 최적화 적용 여부는 탐색중으로 확정되지 않음.

## S19C05 [측정 결과/계획·요청] p63
적용 검토 중인 최적화 기법 후보. 행렬곱 커널 성능 개선을 위해 검토 중인 최적화 기법: (1) 공유 메모리 타일링 - 글로벌 메모리 대역폭 병목 감소 목적, (2) 루프 언롤링(#pragma unroll) - 분기 검사 비용 감소 및 ILP 증대, (3) 고속 내장 수학 함수(-use_fast_math, __expf() 등). 모두 아직 적용되지 않은 검토 대상.

- 한계: 모두 검토 단계이며 실제 적용 및 효과 측정은 이루어지지 않음.

## S19C06 [측정 결과/계획·요청] p63
미시도 최적화 항목. 현재까지 명시적으로 시도하지 않은 최적화 항목: warp divergence 최소화, 비동기 스트림, 블록 사이즈 계산, 메모리 coalescing, 타일링(공유 메모리 타일링으로 위에서 언급됨).

- 한계: 모두 미시도 항목이므로 효과에 대한 실증 데이터 없음.

## S19C07 [측정 결과/계획·요청] p63
배칭 + 다중 스트림 환경 관찰 및 병목 분석. 배칭 + 다중 스트림 관련 실제 환경과의 괴리 검토. 얻은 것: GPU 자원이 남는다는 것, 단일 스트림과 단일 데이터 복사가 현재 큰 병목을 만들고 있다는 것.

- 한계: 실제 환경과의 괴리는 아직 규명되지 않음.

## S20C01 [측정 결과/해석] p66
auto_lirpa backward_only eps sweep 측정 결과. auto_lirpa backward_only 모드에서 eps를 0.0003부터 1.0까지 변화시키며 각 eps별 T, F, ratio, time을 측정. 전체 EPS별 연산 시간 합산은 55.77s, 전체 프로그램 총 소요 시간은 55.78s. backward_only가 forward나 forward+backward보다 빠르게 나와 작성자의 예상(forward < forward+backward < backward_only)과 달랐다.

- T: 31826  (backward_only, eps=0.0003, batch-size 1024, device cuda)
- F: 8174  (backward_only, eps=0.0003, batch-size 1024, device cuda)
- ratio: 0.796  (backward_only, eps=0.0003, batch-size 1024, device cuda)
- time: 2.53s  (backward_only, eps=0.0003, batch-size 1024, device cuda)
- T: 29375  (backward_only, eps=0.0004, batch-size 1024, device cuda)
- F: 10625  (backward_only, eps=0.0004, batch-size 1024, device cuda)
- ratio: 0.734  (backward_only, eps=0.0004, batch-size 1024, device cuda)
- time: 2.52s  (backward_only, eps=0.0004, batch-size 1024, device cuda)
- T: 27170  (backward_only, eps=0.0005, batch-size 1024, device cuda)
- F: 12830  (backward_only, eps=0.0005, batch-size 1024, device cuda)
- ratio: 0.679  (backward_only, eps=0.0005, batch-size 1024, device cuda)
- time: 2.53s  (backward_only, eps=0.0005, batch-size 1024, device cuda)
- T: 25103  (backward_only, eps=0.0006, batch-size 1024, device cuda)
- F: 14897  (backward_only, eps=0.0006, batch-size 1024, device cuda)
- ratio: 0.628  (backward_only, eps=0.0006, batch-size 1024, device cuda)
- time: 2.53s  (backward_only, eps=0.0006, batch-size 1024, device cuda)
- T: 23179  (backward_only, eps=0.0007, batch-size 1024, device cuda)
- F: 16821  (backward_only, eps=0.0007, batch-size 1024, device cuda)
- ratio: 0.579  (backward_only, eps=0.0007, batch-size 1024, device cuda)
- time: 2.59s  (backward_only, eps=0.0007, batch-size 1024, device cuda)
- T: 21323  (backward_only, eps=0.0008, batch-size 1024, device cuda)
- F: 18677  (backward_only, eps=0.0008, batch-size 1024, device cuda)
- ratio: 0.533  (backward_only, eps=0.0008, batch-size 1024, device cuda)
- time: 2.58s  (backward_only, eps=0.0008, batch-size 1024, device cuda)
- T: 19656  (backward_only, eps=0.0009, batch-size 1024, device cuda)
- F: 20344  (backward_only, eps=0.0009, batch-size 1024, device cuda)
- ratio: 0.491  (backward_only, eps=0.0009, batch-size 1024, device cuda)
- time: 2.54s  (backward_only, eps=0.0009, batch-size 1024, device cuda)
- T: 18128  (backward_only, eps=0.001, batch-size 1024, device cuda)
- F: 21872  (backward_only, eps=0.001, batch-size 1024, device cuda)
- ratio: 0.453  (backward_only, eps=0.001, batch-size 1024, device cuda)
- time: 2.57s  (backward_only, eps=0.001, batch-size 1024, device cuda)
- T: 7461  (backward_only, eps=0.002, batch-size 1024, device cuda)
- F: 32539  (backward_only, eps=0.002, batch-size 1024, device cuda)
- ratio: 0.187  (backward_only, eps=0.002, batch-size 1024, device cuda)
- time: 2.51s  (backward_only, eps=0.002, batch-size 1024, device cuda)
- T: 2754  (backward_only, eps=0.003, batch-size 1024, device cuda)
- F: 37246  (backward_only, eps=0.003, batch-size 1024, device cuda)
- ratio: 0.069  (backward_only, eps=0.003, batch-size 1024, device cuda)
- time: 2.51s  (backward_only, eps=0.003, batch-size 1024, device cuda)
- T: 945  (backward_only, eps=0.004, batch-size 1024, device cuda)
- F: 39055  (backward_only, eps=0.004, batch-size 1024, device cuda)
- ratio: 0.024  (backward_only, eps=0.004, batch-size 1024, device cuda)
- time: 2.50s  (backward_only, eps=0.004, batch-size 1024, device cuda)
- T: 286  (backward_only, eps=0.005, batch-size 1024, device cuda)
- F: 39714  (backward_only, eps=0.005, batch-size 1024, device cuda)
- ratio: 0.007  (backward_only, eps=0.005, batch-size 1024, device cuda)
- time: 2.51s  (backward_only, eps=0.005, batch-size 1024, device cuda)
- T: 71  (backward_only, eps=0.006, batch-size 1024, device cuda)
- F: 39929  (backward_only, eps=0.006, batch-size 1024, device cuda)
- ratio: 0.002  (backward_only, eps=0.006, batch-size 1024, device cuda)
- time: 2.50s  (backward_only, eps=0.006, batch-size 1024, device cuda)
- T: 24  (backward_only, eps=0.007, batch-size 1024, device cuda)
- F: 39976  (backward_only, eps=0.007, batch-size 1024, device cuda)
- ratio: 0.001  (backward_only, eps=0.007, batch-size 1024, device cuda)
- time: 2.49s  (backward_only, eps=0.007, batch-size 1024, device cuda)
- T: 3  (backward_only, eps=0.008, batch-size 1024, device cuda)
- F: 39997  (backward_only, eps=0.008, batch-size 1024, device cuda)
- ratio: 0.000  (backward_only, eps=0.008, batch-size 1024, device cuda)
- time: 2.50s  (backward_only, eps=0.008, batch-size 1024, device cuda)
- T: 1  (backward_only, eps=0.009, batch-size 1024, device cuda)
- F: 39999  (backward_only, eps=0.009, batch-size 1024, device cuda)
- ratio: 0.000  (backward_only, eps=0.009, batch-size 1024, device cuda)
- time: 2.52s  (backward_only, eps=0.009, batch-size 1024, device cuda)
- T: 0  (backward_only, eps=0.01, batch-size 1024, device cuda)
- F: 40000  (backward_only, eps=0.01, batch-size 1024, device cuda)
- ratio: 0.000  (backward_only, eps=0.01, batch-size 1024, device cuda)
- time: 2.51s  (backward_only, eps=0.01, batch-size 1024, device cuda)
- T: 0  (backward_only, eps=0.1, batch-size 1024, device cuda)
- F: 40000  (backward_only, eps=0.1, batch-size 1024, device cuda)
- ratio: 0.000  (backward_only, eps=0.1, batch-size 1024, device cuda)
- time: 2.50s  (backward_only, eps=0.1, batch-size 1024, device cuda)
- T: 0  (backward_only, eps=1.0, batch-size 1024, device cuda)
- F: 40000  (backward_only, eps=1.0, batch-size 1024, device cuda)
- ratio: 0.000  (backward_only, eps=1.0, batch-size 1024, device cuda)
- time: 2.48s  (backward_only, eps=1.0, batch-size 1024, device cuda)
- Sum of EPS times: 55.77s  (backward_only, eps 0.0003~1.0 전체, batch-size 1024, device cuda)
- Overall elapsed: 55.78s  (backward_only, eps 0.0003~1.0 전체, batch-size 1024, device cuda)
- 한계: forward 및 forward+backward 모드와의 직접 비교 수치는 본 구간에 없음. backward_only가 가장 빠르다는 관찰은 작성자의 사전 지식/예상과의 비교에 기반함.

## S20C02 [문제·목표/해석] p66,67
배치 사이즈·스트림 구현 방식 관련 문제. 현재 구현은 배치 사이즈만큼 동일한 수의 독립 스트림을 만드는 방식. 배치 사이즈 64일 때 약 424초 소요. 배치 사이즈 1024에서는 1024개 독립 스트림이 각각 GPU 메모리를 차지하여 GPU 메모리 오버 발생. 배치 사이즈 512는 메모리를 넘지 않지만 각 스트림마다 CPU가 다음 커널을 런치해줄 때까지 기다려서 느려짐.

- 소요 시간: 424초 정도  (배치 사이즈 64, auto_lirpa 실행)
- 한계: 배치 사이즈 64의 424초는 figure 캡션 및 메모에 기재된 값으로, 정확한 측정 조건(eps, 모드 등)은 본 구간에서 불명확.

## S20C03 [문제·목표/계획·요청] p66,67
해결 방안 고민 및 계획. 스트림 수를 64개 정도로 줄이고 배치 사이즈를 1024로 하는 방안과 auto_lirpa가 빠른 이유에 대한 질문을 정리. 스트림 수 64개 + 배치 사이즈 1024 방안은 하나의 스트림이 하나의 입력을 처리하므로 16번으로 나눠 처리하게 되어 의미가 없다고 판단. temp_batching 브랜치를 한 번 만져보기로 계획.

- 한계: 방안들은 아직 검증되지 않은 고민 단계이며, 실제 구현·측정 결과는 없음.

## S21C01 [측정 결과/계획·요청] p72
backward_only 성능 측정: CUDA 그래프 적용 및 memcpy 횟수 변경 효과. backward_only 방식에서 CUDA 그래프 적용 여부와 memcpy 횟수 변경(10→6)에 따른 전체 처리 시간 및 bound 시간 측정 결과. 그래프 적용 시 total 787.8832s, bound 658.7972s로 forward(1029.827s)나 forward+backward(1138.49s)보다 빠름.

- total 시간: 2578.377 s (backward_only, N=40000, eps_count=23, memcpy 10번, 그래프 미적용)
- total 시간: 1543.876 s (backward_only, N=40000, eps_count=23, memcpy 6번, 그래프 미적용)
- total 시간: 787.8832 s (backward_only_cuda_graph, N=40000, eps_count=23, memcpy 6번, 그래프 적용)
- total 시간: 1029.827 s (평균(forward))
- total 시간: 1138.49 s (평균(forward+backward))
- bound 시간: 2415.455 s (backward_only, N=40000, eps_count=23, memcpy 10번, 그래프 미적용)
- bound 시간: 1402.504 s (backward_only, N=40000, eps_count=23, memcpy 6번, 그래프 미적용)
- bound 시간: 658.7972 s (backward_only_cuda_graph, N=40000, eps_count=23, memcpy 6번, 그래프 적용)
- bound 시간: 894.5219 s (평균(forward))
- bound 시간: 1003.179 s (평균(forward+backward))
- Elapsed time: 2,1894 ms (CROWN Verification, mempool + warmup, eps calc success, Output Dimension 16, eps=1e-06)
- Inference time: 0.1705 ms (CROWN Verification, mempool + warmup, eps calc success, Output Dimension 16, eps=1e-06)
- Bound time: 2.0184 ms (CROWN Verification, mempool + warmup, eps calc success, Output Dimension 16, eps=1e-06)
- 한계: backward_only가 가장 빠른 이유에 대한 조사는 아직 진행되지 않음(추가 조사 필요).

## S21C02 [측정 결과/해석] p68,69,70
antenna_lirpa(PyTorch) vs C++ 멀티스트림 성능 차이 분석. C++ 멀티스트림(8분) 대비 antenna_lirpa(2분 미만)가 약 4배 빠른 이유에 대한 분석. 스트림 개수가 아니라 연산기(텐서 코어 vs 일반 CUDA 코어)와 커널 호출 방식(3D 텐서 배치 vs 단일 샘플 반복)의 차이로 설명.

- C++ 멀티스트림 소요 시간: 8분 소요 시간 (C++ 멀티스트림(8스트림), Task Parallelism, 단일 샘플 커널을 64개 스트림에 각각 런치, 일반 CUDA 코어(FP32)만 사용, 1024개 처리)
- antenna_lirpa 소요 시간: 2분 미만 소요 시간 (antenna_lirpa(PyTorch), Single Stream, 1024개 입력을 [1024,16,491] 3D 텐서로 병합, 스트림 1개에서 cuBLAS GEMM 1회 호출, Tensor Core 사용, RTX 4070 SUPER)
- 한계: 성능 비교 수치는 이전 확인 결과에 기반한 주장이며, 본 구간의 직접 측정값은 아님.

## S21C03 [측정 결과/추정·가설] p68,71
최적화 방안 논의: 방법 B(커스텀 3D 커널) vs 방법 C(cuBLAS 라이브러리). AI 조언으로 제시된 최적화 방안. 방법 B는 CROWN의 모든 커널에 직접 blockIdx.z를 넣어 3차원 배칭 커널을 작성하는 커스텀 구현, 방법 C는 무거운 행렬 곱셈만 cuBLAS(cublasSgemmBatched)에 맡기는 하이브리드 방식. 로드맵(crown_next_research_roadmap.md)에 방법 A(Multi-Stream 배칭)와 방법 C(Batched GEMM)가 예견되어 있음.

- 한계: 방법 B와 C의 실제 구현 및 성능 측정은 아직 이루어지지 않음. 로드맵의 수치(3.2배, 25분→8분, 30초~1분대)는 계획/목표 수치.

## S21C04 [측정 결과/계획·요청] p72
pinned memory 성능 측정 계획. backward_only가 가장 빠른 이유 조사 및 발표자료 작성을 위해 pinned memory 성능을 정확히 측정하려는 계획. pinned memory 미적용, my_lirpa의 입력 공간 하나(HostAlloc), 현재 방식(HostRegister) 3가지를 측정 예정.

- 한계: 아직 측정되지 않은 계획 단계의 실험. 실제 측정 결과는 없음.

## S22C01 [측정 결과/관측 기록] p73,74
초기 메모리 방식 3종 벤치마크 (클록 고정 전). cudaHostRegister, cudaHostAlloc(h_x0 CPU 복사 포함), No Pinning(Pageable) 세 가지 메모리 방식의 성능을 비교 벤치마크. No Pinning이 전체 실행 시간 776.70초로 가장 빠르고, cudaHostAlloc이 cudaHostRegister보다 8.64초 느림.

- 전체 실행 시간: 787.88s s (cudaHostRegister, backward_only(method=2), CUDA Graph Separate(graph_mode=1), 920,000회)
- 전체 실행 시간: 796.52s s (cudaHostAlloc (h_x0 복사), backward_only(method=2), CUDA Graph Separate(graph_mode=1), 920,000회)
- 전체 실행 시간: 776.70s s (No Pinning (Pageable), backward_only(method=2), CUDA Graph Separate(graph_mode=1), 920,000회)
- 추론 누적 시간: 127.18s s (cudaHostRegister)
- 추론 누적 시간: 130.47s s (cudaHostAlloc (h_x0 복사))
- 추론 누적 시간: 111.92s s (No Pinning (Pageable))
- 바운드 누적 시간: 658.80s s (cudaHostRegister)
- 바운드 누적 시간: 664.46s s (cudaHostAlloc (h_x0 복사))
- 바운드 누적 시간: 663.17s s (No Pinning (Pageable))
- 단일 샘플 평균 지연시간: 약 856 μs μs (cudaHostRegister)
- 단일 샘플 평균 지연시간: 867.43 μs μs (cudaHostAlloc (h_x0 복사))
- 단일 샘플 평균 지연시간: 843.81 μs μs (No Pinning (Pageable))
- P50 중위값: 859.80 μs μs (cudaHostAlloc (h_x0 복사))
- P50 중위값: 837.10 μs μs (No Pinning (Pageable))
- P95 꼬리 지연: 893.80 μs μs (cudaHostAlloc (h_x0 복사))
- P95 꼬리 지연: 867.20 μs μs (No Pinning (Pageable))
- 지터 (StdDev): 24.63 μs μs (cudaHostAlloc (h_x0 복사))
- 지터 (StdDev): 15.95 μs μs (No Pinning (Pageable))
- 순수 GPU 커널 시간: 716.10 μs μs (cudaHostRegister)
- 순수 GPU 커널 시간: 706.34 μs μs (cudaHostAlloc (h_x0 복사))
- 순수 GPU 커널 시간: 716.00 μs μs (No Pinning (Pageable))
- 초당 처리량: 약 1,168 samples/s samples/s (cudaHostRegister)
- 초당 처리량: 1,152.8 samples/s samples/s (cudaHostAlloc (h_x0 복사))
- 초당 처리량: 1,185.1 samples/s samples/s (No Pinning (Pageable))
- 한계: GPU 클록이 고정되지 않은 동적 부스트 상태에서 측정되어 클록 변동에 의한 지터가 포함됨.

## S22C02 [측정 결과/관측 기록] p73
cudaHostAlloc의 h_x0 CPU memcpy 오버헤드 분석. cudaHostAlloc 방식에서 매 루프마다 *pinned_in = dataset[i]; (1,028바이트)를 92만 번 수행하면서 약 8.64초(회당 약 9.4 μs)의 CPU 메모리 복사 지연이 누적됨. 테스트 코드에서 h_x를 사용할 때 불필요한 시뮬레이션용 CPU memcpy 오버헤드가 발생한다는 사실이 수치로 입증됨.

- cudaHostAlloc 추가 지연 (cudaHostRegister 대비): 8.64초 s (cudaHostAlloc (796.52s) - cudaHostRegister (787.88s))
- 회당 CPU 메모리 복사 지연: 약 9.4 μs μs (매 루프 *pinned_in = dataset[i]; 수행 시)
- 복사 대상 크기: 1,028바이트 Byte (dataset[i] → *pinned_in 복사)
- 복사 수행 횟수: 92만 번 회 (총 920,000회 연산 중)
- 한계: h_x0 복사 경로는 시뮬레이션용 테스트 코드에서의 오버헤드이며, 실제 추론 파이프라인에서의 memcpy 비용과는 다를 수 있음.

## S22C03 [측정 결과/관측 기록] p73,74
No Pinning과 Pinned 메모리 간 차이가 작은 이유. 입력 크기가 1KB(256차원 FP32)로 극히 작고, GPU 연산 시간이 전체의 85% 이상을 차지하여 전송 방식 간 전체 실행 시간 차이가 1.4%(약 11초)에 불과함.

- 입력 벡터 크기: 1,024 Byte Byte (256차원 FP32)
- 단일 샘플 처리 시간: 약 840 μs μs (전체 처리 시간)
- GPU 커널 연산 시간: 약 716 μs μs (CROWN 역전파 행렬 계산)
- GPU 커널 연산 시간 비중: 85% 이상 % (CROWN 역전파 행렬 계산)
- 전송 시간 비중: 1~2 μs 수준 μs (전체 처리 시간 대비)
- 한계: 입력 크기가 1KB로 극히 작은 특정 조건에 한정된 해석이며, 수십~수백 MB 단위 전송에서는 DMA 대역폭 차이가 크게 벌어질 수 있음.

## S22C04 [측정 결과/관측 기록] p73,74
cudaHostRegister가 No Pinning보다 느린 원인 분석 (Bound vs Infer 단계). cudaHostRegister가 No Pinning보다 전체 11초(1.4%) 더 느림. Bound 단계(전체의 85%)에서는 Register가 4.37초 더 빨랐으나(DMA 효과), Infer 단계(전체의 15%)에서 Register가 15.26초 더 느려져 역전됨.

- 전체 실행 시간 차이: 11초 s (cudaHostRegister 787.88s vs No Pinning 776.70s)
- 전체 실행 시간 차이 비율: 1.4% % (cudaHostRegister 787.88s vs No Pinning 776.70s)
- 회당 시간 차이: 약 12 μs μs (92만 번 루프 기준)
- Bound 단계 cudaHostRegister: 658.80s s (전체의 85%)
- Bound 단계 No Pinning: 663.17s s (전체의 85%)
- Bound 단계 차이: Register가 4.37초 더 빠름 s (DMA 효과 예상과 일치)
- Infer 단계 cudaHostRegister: 127.18s s (전체의 15%)
- Infer 단계 No Pinning: 111.92s s (전체의 15%)
- Infer 단계 차이: Register가 15.26초 더 느림 s (전체 역전 원인)
- 한계: 3가지 기술적 이유의 구체적 내용은 원문에 기재되지 않아 재현 가능한 수준의 원인 규명은 아님.

## S22C05 [구현·방법/관측 기록] p74,75
GPU 클록 고정 구현 (ScopedGpuClockLock). 일관된 측정을 위해 GPU 클록을 고정하는 RAII 구조체 ScopedGpuClockLock 구현. nvidia-smi -1gc로 클록을 고정하고 소멸자에서 -rgc로 동적 부스트 복원. 기본 클록 2475 MHz로 설정하며, 관리자 권한 필요.

- 한계: 관리자 권한이 필요하며, 환경(OS, GPU 드라이버)에 따라 nvidia-smi 명령의 동작이 다를 수 있음.

## S22C06 [측정 결과/관측 기록] p73,74,75,76
클록 고정 후 메모리 방식 3종 벤치마크 (2800 MHz). GPU 클록을 2800 MHz로 고정한 후 cudaHostRegister, cudaHostAlloc, No Pinning 세 방식의 성능을 재측정. 클록 고정 후에도 No Pinning(790.64s)이 cudaHostAlloc(806.54s)과 cudaHostRegister(798.78s)보다 빠름.

- 전체 실행 시간: 798.78s s (cudaHostRegister, 클록 2800MHz 고정, backward_only(method=2), graph_mode=1, 100,000 샘플)
- 전체 실행 시간: 806.54s s (cudaHostAlloc, 클록 2800MHz 고정, backward_only(method=2), graph_mode=1, 100,000 샘플)
- 전체 실행 시간: 790.64s s (No Pinning, 클록 2800MHz 고정, backward_only(method=2), graph_mode=1, 100,000 샘플)
- 추론 누적 시간: 128,893s s (cudaHostRegister, 클록 2800MHz)
- 추론 누적 시간 비중: 16.1% % (cudaHostRegister, 클록 2800MHz)
- 추론 누적 시간: 132,586s s (cudaHostAlloc, 클록 2800MHz)
- 추론 누적 시간 비중: 16.4% % (cudaHostAlloc, 클록 2800MHz)
- 추론 누적 시간: 119,466s s (No Pinning, 클록 2800MHz)
- 추론 누적 시간 비중: 15.1% % (No Pinning, 클록 2800MHz)
- 바운드 누적 시간: 667.159s s (cudaHostRegister, 클록 2800MHz)
- 바운드 누적 시간 비중: 83.5% % (cudaHostRegister, 클록 2800MHz)
- 바운드 누적 시간: 670.122s s (cudaHostAlloc, 클록 2800MHz)
- 바운드 누적 시간 비중: 83.1% % (cudaHostAlloc, 클록 2800MHz)
- 바운드 누적 시간: 667.611s s (No Pinning, 클록 2800MHz)
- 바운드 누적 시간 비중: 84.4% % (No Pinning, 클록 2800MHz)
- certify 시간: 1,644s s (cudaHostRegister, 클록 2800MHz)
- certify 시간 비중: 0.2% % (cudaHostRegister, 클록 2800MHz)
- certify 시간: 2,077s s (cudaHostAlloc, 클록 2800MHz)
- certify 시간 비중: 0.3% % (cudaHostAlloc, 클록 2800MHz)
- certify 시간: 2,004s s (No Pinning, 클록 2800MHz)
- certify 시간 비중: 0.3% % (No Pinning, 클록 2800MHz)
- 평균 지연시간: 873.54 us us (cudaHostRegister, 클록 2800MHz)
- 평균 지연시간: 869.34 us us (cudaHostAlloc, 클록 2800MHz)
- 평균 지연시간: 859.20 us us (No Pinning, 클록 2800MHz)
- P50 중위값: 860.10 us us (cudaHostRegister, 클록 2800MHz)
- P50 중위값: 859.80 us us (cudaHostAlloc, 클록 2800MHz)
- P50 중위값: 847.90 us us (No Pinning, 클록 2800MHz)
- P95 꼬리 지연: 971.40 us us (cudaHostRegister, 클록 2800MHz)
- P95 꼬리 지연: 899.60 us us (cudaHostAlloc, 클록 2800MHz)
- P95 꼬리 지연: 899.40 us us (No Pinning, 클록 2800MHz)
- P99 꼬리 지연: 1087.80 us us (cudaHostRegister, 클록 2800MHz)
- P99 꼬리 지연: 1095.90 us us (cudaHostAlloc, 클록 2800MHz)
- P99 꼬리 지연: 1089.30 us us (No Pinning, 클록 2800MHz)
- 최대 지연시간: 10943.50 us us (cudaHostRegister, 클록 2800MHz)
- 최대 지연시간: 13996.40 us us (cudaHostAlloc, 클록 2800MHz)
- 최대 지연시간: 14400,40 us us (No Pinning, 클록 2800MHz)
- 지터 (StdDev): 77.94 us us (cudaHostRegister, 클록 2800MHz)
- 지터 (StdDev): 73.27 us us (cudaHostAlloc, 클록 2800MHz)
- 지터 (StdDev): 105.84 us us (No Pinning, 클록 2800MHz)
- 처리량: 1144.77 samples/sec samples/sec (cudaHostRegister, 클록 2800MHz)
- 처리량: 1150.30 samples/sec samples/sec (cudaHostAlloc, 클록 2800MHz)
- 처리량: 1163.87 samples/sec samples/sec (No Pinning, 클록 2800MHz)
- 순수 GPU 실행 시간: 717.09 us us (cudaHostRegister, 클록 2800MHz, 마지막 샘플)
- 순수 GPU 실행 시간: 705.89 us us (cudaHostAlloc, 클록 2800MHz, 마지막 샘플)
- 순수 GPU 실행 시간: 695.42 us us (No Pinning, 클록 2800MHz, 마지막 샘플)
- ⚠ 소수 쉼표 의심: 원문 '128,893s'는 128.893s일 가능성. 근거: 천 단위로 읽으면 128893s가 total elapsed 798.78s보다 큼; 소수로 읽은 128.893s가 total 대비 16.1%와 일치
- ⚠ 소수 쉼표 의심: 원문 '132,586s'는 132.586s일 가능성. 근거: 천 단위로 읽으면 132586s가 total elapsed 806.54s보다 큼; 소수로 읽은 132.586s가 total 대비 16.4%와 일치
- ⚠ 소수 쉼표 의심: 원문 '119,466s'는 119.466s일 가능성. 근거: 천 단위로 읽으면 119466s가 total elapsed 790.64s보다 큼; 소수로 읽은 119.466s가 total 대비 15.1%와 일치
- ⚠ 소수 쉼표 의심: 원문 '1,644s'는 1.644s일 가능성. 근거: 천 단위로 읽으면 1644s가 total elapsed 798.78s보다 큼; 소수로 읽은 1.644s가 total 대비 0.2%와 일치
- ⚠ 소수 쉼표 의심: 원문 '2,077s'는 2.077s일 가능성. 근거: 천 단위로 읽으면 2077s가 total elapsed 806.54s보다 큼; 소수로 읽은 2.077s가 total 대비 0.3%와 일치
- ⚠ 소수 쉼표 의심: 원문 '2,004s'는 2.004s일 가능성. 근거: 천 단위로 읽으면 2004s가 total elapsed 790.64s보다 큼; 소수로 읽은 2.004s가 total 대비 0.3%와 일치
- ⚠ 서술 수치가 원문 구간에 없음: 2800,2800
- 한계: 단일 클록(2800 MHz)에서만 측정되었으며, 다른 클록 값에서의 경향은 확인되지 않음. Max Latency에서 No Pinning(14400,40 us)이 cudaHostAlloc(13996.40 us)보다 큰 이상치가 존재함.

## S22C07 [구현·방법/관측 기록] p75
GPU Throttling 원인과 대응 방안. GPU 클록 저하(Throttling)의 세 가지 원인(온도, 전력, API 간헐적 지연)과 증상, 해결책을 정리. 일관된 측정을 위한 설정으로 Persistence mode 활성화(sudo nvidia-smi -pm 1)를 제시.

- 한계: 표는 일반적 Throttling 원인 정리이며, 본 벤치마크 환경에서의 구체적 발생 여부는 별도 측정 필요.

## S22C08 [문제·목표/계획·요청] p75,76
클록 고정 후에도 No Pinning이 더 빠른 현상에 대한 미해결 질문. GPU 클록을 2800 MHz로 고정하고 측정했음에도 No Pinning이 cudaHostAlloc, cudaHostRegister보다 빠른 결과가 나와, 이 현상의 원인에 대한 추가 설명(Gemini 답변)을 요청한 상태. 원인은 아직 규명되지 않음.

- 한계: 원인이 규명되지 않아 논문 고찰에서 확정적 진술로 사용할 수 없음. 후속 분석 결과에 따라 내용 변경 가능.

## S23C01 [측정 결과/해석] p77,80
1 KB 데이터에서 No-Pin이 Pinned보다 빠른 역설 현상. 1 KB(1,028 Byte) 크기의 초소형 데이터 전송에서 No-Pin(일반 페이지블 메모리)이 Pinned Memory보다 총 소요 시간이 더 짧게 측정된 역설적 현상. 이는 AI 조언에 따르면 CUDA 교과서의 "Pinned Memory가 무조건 빠르다"는 명제에 "데이터가 충분히 클 때"라는 숨은 전제가 있기 때문에 발생하는 "초소형 데이터(1 KB)의 역설"이다.

- Pinned infer 단계 총 소요 시간: 132 초 (1,028 Byte 입력, 92만 번 루프 반복, Pinned vs No-Pin 비교, DMA 셋업 오버헤드 누적)
- No-Pin infer 단계 총 소요 시간: 119 초 (1,028 Byte 입력, 92만 번 루프 반복, Pinned vs No-Pin 비교, DMA 셋업 오버헤드 누적)
- 한계: 이 현상은 초소형 데이터(1 KB)에 국한되며, 데이터가 충분히 클 때는 Pinned Memory가 더 빠름

## S23C02 [측정 결과/해석] p77,79,80
매 루프 DMA 셋업 오버헤드 누적 계산 (920,000회). 1,028 Byte 데이터를 920,000회 반복 전송할 때 1회 전송 오버헤드가 누적되어 총 오버헤드로 환산된 값. AI 조언에 따르면 No-Pin은 CPU 캐시 복사로 오버헤드가 극히 작고(~0.03초), Pinned는 하드웨어 DMA 셋업 오버헤드가 누적되어 약 1.38초에 달함.

- No-Pin(CPU 캐시 복사) 1회 전송 오버헤드: ~0.03 μs (No-Pin(일반 메모리), 1회 전송 기준)
- No-Pin(CPU 캐시 복사) 920,000회 누적 오버헤드: ~0.03 초 (No-Pin(일반 메모리), 920,000회 누적 기준)
- Pinned(하드웨어 DMA 셋업) 1회 전송 오버헤드: ~1.50 μs (Pinned(하드웨어 DMA 셋업), 1회 전송 기준)
- Pinned(하드웨어 DMA 셋업) 920,000회 누적 오버헤드: ~1.38 초 (Pinned(하드웨어 DMA 셋업), 920,000회 누적 기준)
- 한계: 1회 오버헤드 값(~0.03 μs, ~1.50 μs)은 AI 추정/계산값이며 직접 측정된 값은 아님

## S23C03 [측정 결과/해석] p78,80
실시간성 지표 비교 - 최대 지연시간(WCET)과 지터. 평균 처리 속도에서는 No-Pin이 더 빠르지만, 실시간성 지표인 최대 지연시간(Max Spike/WCET)과 지터(Jitter)에서는 Pinned Memory가 압도적으로 안정적임을 보인 비교. AI 조언에 따르면 No-Pin은 OS 페이징 간섭으로 치명적 스파이크(최대 11.6ms)가 발생하고, Pinned는 결정론적 성능(최대 1.64ms)을 보장함.

- No-Pin 최대 지연(Max Spike): 8.8 ~ 11.6 ms (No-Pin(일반 메모리), OS 페이징/백그라운드 작업 간섭 시)
- Pinned 최대 지연(Max Spike): 1.64 ms (Pinned Memory(Register), 모든 상황)
- No-Pin 지터(Jitter): 118 μs (No-Pin(일반 메모리))
- Pinned 지터(Jitter): 47 μs (Pinned Memory(Register))
- No-Pin 최대 지연시간(Max): 8.80 ~ 11.6 ms (No-Pin(일반 RAM))
- Pinned 최대 지연시간(Max): 1.64 ms (Pinned(cudaHostRegister))
- No-Pin 지터(표준편차): 118.94 μs (No-Pin(일반 RAM))
- Pinned 지터(표준편차): 47.28 μs (Pinned(cudaHostRegister))
- 한계: 지터 값 118 μs(p78)와 118.94 μs(p80), 47 μs(p78)와 47.28 μs(p80)는 원문 내 표기 차이가 있음

## S23C04 [측정 결과/해석] p78
대용량 데이터에서 Pinned Memory의 이점. 데이터가 수 MB ~ 수백 MB로 커지면 CPU 복사 속도가 감당이 안 되고 DMA의 고속 대역폭이 압도적이 되어 Pinned Memory가 일반 메모리보다 훨씬 빨라진다는 설명. AI 조언에 따르면 이 조건에서 Pinned Memory가 23배 빠르며, DMA 대역폭은 1224GB/s임.

- Pinned Memory DMA 고속 대역폭: 1224 GB/s (Pinned Memory 사용 시, 대용량 데이터 조건)
- 대용량 데이터에서 Pinned vs 일반 메모리 속도 차이: 23 배 (데이터가 수 MB ~ 수백 MB일 때, Pinned Memory vs 일반 메모리)
- 한계: 1224GB/s와 23배는 AI 조언에서 인용된 값으로, 본 실험에서 직접 측정된 값은 아님

## S23C05 [측정 결과/해석] p78
CUDA 공식 가이드 - Pageable 메모리에서 cudaMemcpyAsync 동작. NVIDIA 공식 CUDA 프로그래밍 가이드에 명시된 내용으로, 호스트 메모리가 Pageable(일반 메모리)일 때 cudaMemcpyAsync는 비동기가 아니라 CPU가 드라이버의 내부 고정 스테이징 버퍼로 동기식 memcpy를 즉시 수행한다는 기술적 사실. AI 조언에 따르면 이로 인해 No-Pin일 때 CPU는 1KB를 L1 캐시 복사 속도로 스테이징 버퍼에 복사하고 바로 다음 코드로 넘어가며, 드라이버 스테이징 버퍼는 GPU 메모리 매퍼에 영구 등록된 고정 주소라서 GPU TLB 미스도 발생하지 않음.

- 한계: 가이드 인용 이후의 메커니즘 설명(L1 캐시 복사, TLB 미스 없음 등)은 AI 추정

## S23C06 [측정 결과/해석] p77,79,80
매 루프마다 발생하는 하드웨어 동작 메커니즘 (Pinned vs No-Pin). cudaMemcpyAsync가 CUDA Graph 바깥에서 매 반복문마다 개별 호출될 때, Pinned Memory와 No-Pin(일반 메모리)의 드라이버 내부 동작이 완전히 다르다는 메커니즘 설명. AI 조언에 따르면 Pinned는 DMA 디스크립터 생성 및 하드웨어 큐 제출 오버헤드가 호출 1회당 약 1.0~1.5 μs 소모되고, No-Pin은 CPU L1/L2 캐시로 드라이버 내부 임시 버퍼에 memcpy하여 0.02~0.05 μs(20~50 나노초)면 끝남.

- Pinned DMA 디스크립터 생성 및 하드웨어 큐 제출 오버헤드 (1회당): 1.0 ~ 1.5 μs (Pinned Memory(cudaHostRegister/cudaHostAlloc), 호출 1회당, DMA 디스크립터 패킷 생성 및 MMIO 큐 제출 포함)
- No-Pin CPU L1/L2 캐시 memcpy 시간 (1,028 바이트): 0.02 ~ 0.05 μs (No-Pin(일반 Pageable 메모리), CPU L1/L2 캐시에서 드라이버 내부 임시 버퍼로 memcpy)
- No-Pin CPU L1 캐시 복사 시간 (1,028 Byte): 20~30 나노초 (No-Pin, CPU L1 캐시에서 드라이버 내부 버퍼로 복사)
- Pinned PCle DMA 엔진 Setup 하드웨어 오버헤드: 1 ~ 2 마이크로초 (Pinned Memory, PCle DMA 엔진에 명령 패킷을 쏘고 트랜잭션 시작(Setup))
- 한계: 오버헤드 값들은 AI 추정/계산값이며, 실제 하드웨어 측정값은 아님. 1~2 μs(p77)와 1.0~1.5 μs(p79) 간 표기 차이가 있음.

## S24C01 [측정 결과/관측 기록] p81
작은 전송(1 KiB)에서 pinned memory 미적용 버전이 더 빠른 현상. 여러 번 측정 결과, 1 KiB 입력 기준 pinned memory를 미적용한 버전의 복사 속도가 약 3배 빠르다고 언급되었다(p81:e624). 추론 종료까지 동기화(synchronize) 시간도 pinned 미적용 버전이 가장 빠르다(p81:e625). cudaMemcpyAsync 호출 시간은 pinned 적용 버전이 더 빠르거나 느렸다로 측정마다 방향이 달라 일관되지 않는다(p81:e626, p81:e628). 저자는 작은 데이터 크기를 staging buffer로 옮기는 고정 비용이 크지 않을 수 있다고 언급했다(p81:e627).

- cuda stream synchronize 평균값 (noPinning): 314μs μs (n=920000, 1 KiB 입력, 평균값, noPinning 버전)
- cuda stream synchronize 평균값 (hostRegister): 323μs μs (n=920000, 1 KiB 입력, 평균값, hostRegister 버전)
- cuda stream synchronize 평균값 (hostAlloc): 322μs μs (n=920000, 1 KiB 입력, 평균값, hostAlloc 버전)
- cudaMemcpyAsync 평균값 (noPinning): 4.932μs μs (n=920000, 1 KiB 입력, 평균값, noPinning 버전)
- cudaMemcpyAsync 평균값 (hostRegister): 5.188μs μs (n=920000, 1 KiB 입력, 평균값, hostRegister 버전)
- cudaMemcpyAsync 평균값 (hostAlloc): 5.351μs μs (n=920000, 1 KiB 입력, 평균값, hostAlloc 버전)
- memcpy 복사 시간 평균값 (noPinning): 340ns ns (n=920000, 1 KiB 입력, 평균값, noPinning 버전)
- memcpy 복사 시간 평균값 (hostRegister): 1.117μs μs (n=920000, 1 KiB 입력, 평균값, hostRegister 버전)
- memcpy 복사 시간 평균값 (hostAlloc): 1.142μs μs (n=920000, 1 KiB 입력, 평균값, hostAlloc 버전)
- 한계: CUDA 내부 구현 직접 확인 불가로 원인 확정 불가

## S24C02 [측정 결과/관측 기록] p82
CROWN 3가지 방법의 커널 퓨전 및 backward_only 메모리 복사 커널 제거에 따른 커널 발사 횟수 감소. 3-layer 네트워크 기준, 하나의 bound 내에서 H2D/D2H memcpy, cudaStreamSynchronize, cudaGraphLaunch를 제외하고 CUDA kernel launch만 계수했을 때 Forward method=0은 27→18회(9회 감소), Backward method=1은 46→31회(15회 감소), Backward-only method=2는 45→33회(12회 감소)했다(p82:enh:e3). 그래프 모드 1에서 샘플 1개 전체 기준으로는 Forward 33→24회, Backward 52→37회, Backward-only 51→39회 감소했다(p82:enh:e5). 추론 커널은 그대로 유지되었다(p82:enh:e4).

- Forward method=0 bound 커널 횟수 감소: 9회 회 (기존 27회, 현재 18회, 3-layer 네트워크, bound 내 계수, H2D/D2H memcpy·cudaStreamSynchronize·cudaGraphLaunch 제외)
- Backward method=1 bound 전체 커널 횟수 감소: 15회 회 (기존 46회, 현재 31회, 3-layer 네트워크, bound 내 계수, H2D/D2H memcpy·cudaStreamSynchronize·cudaGraphLaunch 제외)
- Backward-only method=2 bound 전체 커널 횟수 감소: 12회 회 (기존 45회, 현재 33회, 3-layer 네트워크, bound 내 계수, H2D/D2H memcpy·cudaStreamSynchronize·cudaGraphLaunch 제외)
- Forward 전체 커널 횟수 감소 (샘플 1개): 9회 회 (기존 33회, 현재 24회, graph mode=1, 샘플 1개 전체 기준, H2D/D2H memcpy·cudaStreamSynchronize·cudaGraphLaunch 제외)
- Backward 전체 커널 횟수 감소 (샘플 1개): 15회 회 (기존 52회, 현재 37회, graph mode=1, 샘플 1개 전체 기준, H2D/D2H memcpy·cudaStreamSynchronize·cudaGraphLaunch 제외)
- Backward-only 전체 커널 횟수 감소 (샘플 1개): 12회 회 (기존 51회, 현재 39회, graph mode=1, 샘플 1개 전체 기준, H2D/D2H memcpy·cudaStreamSynchronize·cudaGraphLaunch 제외)
- 한계: 추론 커널은 그대로 유지됨 (p82:enh:e4)

## S24C03 [측정 결과/관측 기록] p83,84
커널 퓨전 적용 전후 전체 elapsed 시간 비교. 커널 퓨전 및 backward_only 메모리 복사 커널 제거 적용 후 전체 elapsed 시간이 감소했다. backward_only는 787s → 630s(157초 정도 감소, p83:e639, p83:review:e3), forward는 1029s → 668s(p83:e642), forward+backward는 1138s → 758s(p84:e645)로 줄었다. 정밀 측정값 기준 total elapsed는 backward_only 630.94s(p83:e638), forward 668.37s(p83:e641, p83:review:e5), forward+backward 758.37s(p84:e644)로 측정되었다.

- backward_only total elapsed (이전, 근사): 787s s (all eps × all data, 캡션 기준)
- backward_only total elapsed (이후, 근사): 630s s (all eps × all data, 캡션 기준)
- backward_only total elapsed (이후, 정밀): 630.94s s (all eps × all data, figure 기준)
- backward_only 감소량: 157초 정도 초 (all eps × all data, 캡션 기준)
- forward total elapsed (이전, 근사): 1029s s (all eps × all data, 캡션 기준)
- forward total elapsed (이후, 근사): 668s s (all eps × all data, 캡션 기준)
- forward total elapsed (이후, 정밀): 668.37s s (all eps × all data, figure 기준)
- forward total elapsed (이후, 정밀, 검토): 668.37s s (all eps × all data, review 기준)
- forward+backward total elapsed (이전, 근사): 1138s s (all eps × all data, 캡션 기준)
- forward+backward total elapsed (이후, 근사): 758s s (all eps × all data, 캡션 기준)
- forward+backward total elapsed (이후, 정밀): 758.37s s (all eps × all data, figure 기준)
- 한계: 각 방법의 측정 조건이 완전히 동일한지 확인 필요

## S24C04 [측정 결과/관측 기록] p83
Real-Time Single-Sample(B=1) 지연 시간 및 처리량 측정. Real-Time Single-Sample(B=1) Latency Analysis 결과, backward_only_cuda_graph 기준 100,000개 샘플 평가 시 평균 지연 689.64μs(0.69ms), 중앙값(P50) 682.00μs, P95 꼬리 지연 712.20μs, P99 꼬리 지연 822.90μs, 최대 지연 13511.00μs, 지터(StdDev) 76.80μs, 처리량 1450.02 samples/sec, Pure GPU Execution 522.05μs로 측정되었다(p83:e638). forward 기준 평균 지연 723.97μs(0.72ms), 중앙값 719.80μs, P95 752.50μs, P99 780.90μs, 최대 5496.50μs, 지터 28.04μs, 처리량 1381.28 samples/sec, Pure GPU Execution 570.08μs로 측정되었다(p83:e641, p83:review:e5). total elapsed는 backward_only 630.94s, forward 668.37s이다.

- total elapsed (backward_only): 630.94s s (all eps × all data, figure 기준)
- total elapsed (forward): 668.37s s (all eps × all data, figure 기준)
- total elapsed (forward, 검토): 668.37s s (all eps × all data, review 기준)
- Mean Latency (backward_only): 689.64μs μs (backward_only_cuda_graph, B=1, n=100000)
- Mean Latency (backward_only, ms 표기): 0.69ms ms (backward_only_cuda_graph, B=1, n=100000)
- Median (P50) Latency (backward_only): 682.00μs μs (backward_only_cuda_graph, B=1, n=100000)
- P95 Tail Latency (backward_only): 712.20μs μs (backward_only_cuda_graph, B=1, n=100000)
- P99 Tail Latency (backward_only): 822.90μs μs (backward_only_cuda_graph, B=1, n=100000)
- Max Latency (backward_only): 13511.00μs μs (backward_only_cuda_graph, B=1, n=100000)
- Jitter (StdDev) (backward_only): 76.80μs μs (backward_only_cuda_graph, B=1, n=100000)
- Throughput (backward_only): 1450.02 samples/sec samples/sec (backward_only_cuda_graph, B=1, n=100000)
- Pure GPU Execution (backward_only): 522.05μs μs (backward_only_cuda_graph, B=1, n=100000, last sample)
- Mean Latency (forward): 723.97μs μs (forward, B=1, n=100000)
- Mean Latency (forward, ms 표기): 0.72ms ms (forward, B=1, n=100000)
- Median (P50) Latency (forward): 719.80μs μs (forward, B=1, n=100000)
- P95 Tail Latency (forward): 752.50μs μs (forward, B=1, n=100000)
- P99 Tail Latency (forward): 780.90μs μs (forward, B=1, n=100000)
- Max Latency (forward): 5496.50μs μs (forward, B=1, n=100000)
- Jitter (StdDev) (forward): 28.04μs μs (forward, B=1, n=100000)
- Throughput (forward): 1381.28 samples/sec samples/sec (forward, B=1, n=100000)
- Pure GPU Execution (forward): 570.08μs μs (forward, B=1, n=100000, last sample)
- 한계: 두 방법의 측정 조건이 완전히 동일한지 확인 필요

## S24C05 [측정 결과/해석] p83,84
커널 횟수 감소와 시간 감소 폭의 불일치 분석. Forward에서 커널 호출 횟수가 9회 줄어 가장 많이 감소했으나, 실제 시간 감소는 forward(1029s→668s, p83:e642)가 backward_only(787s→630s, p83:e639)보다 크다. 저자는 forward에서 무거운 커널 연산의 호출 횟수가 9개 줄어든 반면, backward에서는 forward 부분 12개(9+3, cache 저장 커널 3개 추가 감소), 실제 backward 부분은 3개만 감소했고, backward_only는 커널 감소 비율이 26.7%(45→33회, p84:e647)로 가장 낮으며 줄어든 커널도 단순 가중치·bias 복사 및 부호 분리 커널이라 커널 비용이 다르기 때문으로 추정한다(p84:e647).

- Forward 커널 감소 횟수 (무거운 커널 연산): 9회 회 (커널 퓨전 적용 후, forward 방법)
- Backward forward 부분 커널 감소 횟수: 12회 회 (커널 퓨전 적용 후, backward 방법, forward 부분 (9+3, cache 저장 커널 3개 포함))
- Backward 실제 backward 부분 커널 감소 횟수: 3회 회 (커널 퓨전 적용 후, backward 방법, 실제 backward 부분)
- Backward-only 커널 감소 비율: 26.7% % (커널 퓨전 적용 후, 45→33회, 가장 낮은 감소 비율)
- Backward-only 가중치·bias 복사 커널 감소 횟수: 6회 회 (커널 퓨전 적용 후, 단순 가중치·bias 복사 커널)
- Backward-only 부호 분리 커널 감소 횟수: 6회 회 (커널 퓨전 적용 후, 부호 분리 커널)
- forward total elapsed (이전): 1029s s (all eps × all data, 캡션 기준)
- forward total elapsed (이후): 668s s (all eps × all data, 캡션 기준)
- backward_only total elapsed (이후): 630s s (all eps × all data, 캡션 기준)
- 한계: 커널별 실행 비용 정량 측정 없이 추정에 기반함

## S24C06 [측정 결과/관측 기록] p84
진행 사항: pinned memory 입력층 제거. pinned memory 입력층은 제거했으나, Graph manager에는 필요에 따라 pinned memory를 선택 할당 가능하도록 남겨 두었다.

- 한계: Graph manager의 pinned memory 선택 할당 기능이 실제 사용되는지 여부는 별도 확인 필요
