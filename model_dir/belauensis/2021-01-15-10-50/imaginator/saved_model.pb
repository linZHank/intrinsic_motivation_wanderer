��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.12v2.3.0-54-gfcc4b966f18��
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

: *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
: *
dtype0
�
imagined_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_nameimagined_mean/kernel
}
(imagined_mean/kernel/Read/ReadVariableOpReadVariableOpimagined_mean/kernel*
_output_shapes

: *
dtype0
|
imagined_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameimagined_mean/bias
u
&imagined_mean/bias/Read/ReadVariableOpReadVariableOpimagined_mean/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
trainable_variables
regularization_losses
	variables
		keras_api


signatures
 
 
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api

0
1
2
3
 

0
1
2
3
�
trainable_variables
regularization_losses

layers
non_trainable_variables
layer_metrics
	variables
metrics
layer_regularization_losses
 
 
 
 
�
trainable_variables
regularization_losses

 layers
!non_trainable_variables
"layer_metrics
	variables
#metrics
$layer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
trainable_variables
regularization_losses

%layers
&non_trainable_variables
'layer_metrics
	variables
(metrics
)layer_regularization_losses
`^
VARIABLE_VALUEimagined_mean/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEimagined_mean/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
trainable_variables
regularization_losses

*layers
+non_trainable_variables
,layer_metrics
	variables
-metrics
.layer_regularization_losses
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
$serving_default_imaginator_input_actPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
&serving_default_imaginator_input_statePlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCall$serving_default_imaginator_input_act&serving_default_imaginator_input_statedense_1/kerneldense_1/biasimagined_mean/kernelimagined_mean/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *+
f&R$
"__inference_signature_wrapper_2574
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp(imagined_mean/kernel/Read/ReadVariableOp&imagined_mean/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *&
f!R
__inference__traced_save_2730
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/biasimagined_mean/kernelimagined_mean/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *)
f$R"
 __inference__traced_restore_2752�
�
�
+__inference_functional_5_layer_call_fn_2628
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_functional_5_layer_call_and_return_conditional_losses_25172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
F__inference_functional_5_layer_call_and_return_conditional_losses_2481
imaginator_input_state
imaginator_input_act
dense_1_2449
dense_1_2451
imagined_mean_2475
imagined_mean_2477
identity��dense_1/StatefulPartitionedCall�%imagined_mean/StatefulPartitionedCall�
concatenate/PartitionedCallPartitionedCallimaginator_input_stateimaginator_input_act*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_24182
concatenate/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1_2449dense_1_2451*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_24382!
dense_1/StatefulPartitionedCall�
%imagined_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0imagined_mean_2475imagined_mean_2477*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_imagined_mean_layer_call_and_return_conditional_losses_24642'
%imagined_mean/StatefulPartitionedCall�
IdentityIdentity.imagined_mean/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall&^imagined_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������:���������::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2N
%imagined_mean/StatefulPartitionedCall%imagined_mean/StatefulPartitionedCall:_ [
'
_output_shapes
:���������
0
_user_specified_nameimaginator_input_state:]Y
'
_output_shapes
:���������
.
_user_specified_nameimaginator_input_act
�
�
G__inference_imagined_mean_layer_call_and_return_conditional_losses_2464

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :::O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
__inference__traced_save_2730
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop3
/savev2_imagined_mean_kernel_read_readvariableop1
-savev2_imagined_mean_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1964d17922034e45b211053d44224317/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop/savev2_imagined_mean_kernel_read_readvariableop-savev2_imagined_mean_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*7
_input_shapes&
$: : : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
�
q
E__inference_concatenate_layer_call_and_return_conditional_losses_2649
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
G__inference_imagined_mean_layer_call_and_return_conditional_losses_2685

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :::O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
F__inference_functional_5_layer_call_and_return_conditional_losses_2547

inputs
inputs_1
dense_1_2536
dense_1_2538
imagined_mean_2541
imagined_mean_2543
identity��dense_1/StatefulPartitionedCall�%imagined_mean/StatefulPartitionedCall�
concatenate/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_24182
concatenate/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1_2536dense_1_2538*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_24382!
dense_1/StatefulPartitionedCall�
%imagined_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0imagined_mean_2541imagined_mean_2543*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_imagined_mean_layer_call_and_return_conditional_losses_24642'
%imagined_mean/StatefulPartitionedCall�
IdentityIdentity.imagined_mean/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall&^imagined_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������:���������::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2N
%imagined_mean/StatefulPartitionedCall%imagined_mean/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
V
*__inference_concatenate_layer_call_fn_2655
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_24182
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
{
&__inference_dense_1_layer_call_fn_2675

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_24382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference__wrapped_model_2406
imaginator_input_state
imaginator_input_act7
3functional_5_dense_1_matmul_readvariableop_resource8
4functional_5_dense_1_biasadd_readvariableop_resource=
9functional_5_imagined_mean_matmul_readvariableop_resource>
:functional_5_imagined_mean_biasadd_readvariableop_resource
identity��
$functional_5/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$functional_5/concatenate/concat/axis�
functional_5/concatenate/concatConcatV2imaginator_input_stateimaginator_input_act-functional_5/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2!
functional_5/concatenate/concat�
*functional_5/dense_1/MatMul/ReadVariableOpReadVariableOp3functional_5_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*functional_5/dense_1/MatMul/ReadVariableOp�
functional_5/dense_1/MatMulMatMul(functional_5/concatenate/concat:output:02functional_5/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
functional_5/dense_1/MatMul�
+functional_5/dense_1/BiasAdd/ReadVariableOpReadVariableOp4functional_5_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+functional_5/dense_1/BiasAdd/ReadVariableOp�
functional_5/dense_1/BiasAddBiasAdd%functional_5/dense_1/MatMul:product:03functional_5/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
functional_5/dense_1/BiasAdd�
functional_5/dense_1/ReluRelu%functional_5/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
functional_5/dense_1/Relu�
0functional_5/imagined_mean/MatMul/ReadVariableOpReadVariableOp9functional_5_imagined_mean_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0functional_5/imagined_mean/MatMul/ReadVariableOp�
!functional_5/imagined_mean/MatMulMatMul'functional_5/dense_1/Relu:activations:08functional_5/imagined_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2#
!functional_5/imagined_mean/MatMul�
1functional_5/imagined_mean/BiasAdd/ReadVariableOpReadVariableOp:functional_5_imagined_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1functional_5/imagined_mean/BiasAdd/ReadVariableOp�
"functional_5/imagined_mean/BiasAddBiasAdd+functional_5/imagined_mean/MatMul:product:09functional_5/imagined_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2$
"functional_5/imagined_mean/BiasAdd
IdentityIdentity+functional_5/imagined_mean/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������:���������:::::_ [
'
_output_shapes
:���������
0
_user_specified_nameimaginator_input_state:]Y
'
_output_shapes
:���������
.
_user_specified_nameimaginator_input_act
�
�
F__inference_functional_5_layer_call_and_return_conditional_losses_2594
inputs_0
inputs_1*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource0
,imagined_mean_matmul_readvariableop_resource1
-imagined_mean_biasadd_readvariableop_resource
identity�t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis�
concatenate/concatConcatV2inputs_0inputs_1 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatenate/concat�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulconcatenate/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_1/Relu�
#imagined_mean/MatMul/ReadVariableOpReadVariableOp,imagined_mean_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#imagined_mean/MatMul/ReadVariableOp�
imagined_mean/MatMulMatMuldense_1/Relu:activations:0+imagined_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
imagined_mean/MatMul�
$imagined_mean/BiasAdd/ReadVariableOpReadVariableOp-imagined_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$imagined_mean/BiasAdd/ReadVariableOp�
imagined_mean/BiasAddBiasAddimagined_mean/MatMul:product:0,imagined_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
imagined_mean/BiasAddr
IdentityIdentityimagined_mean/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������:���������:::::Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
"__inference_signature_wrapper_2574
imaginator_input_act
imaginator_input_state
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallimaginator_input_stateimaginator_input_actunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *(
f#R!
__inference__wrapped_model_24062
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
'
_output_shapes
:���������
.
_user_specified_nameimaginator_input_act:_[
'
_output_shapes
:���������
0
_user_specified_nameimaginator_input_state
�
�
F__inference_functional_5_layer_call_and_return_conditional_losses_2614
inputs_0
inputs_1*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource0
,imagined_mean_matmul_readvariableop_resource1
-imagined_mean_biasadd_readvariableop_resource
identity�t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis�
concatenate/concatConcatV2inputs_0inputs_1 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatenate/concat�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulconcatenate/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_1/Relu�
#imagined_mean/MatMul/ReadVariableOpReadVariableOp,imagined_mean_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#imagined_mean/MatMul/ReadVariableOp�
imagined_mean/MatMulMatMuldense_1/Relu:activations:0+imagined_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
imagined_mean/MatMul�
$imagined_mean/BiasAdd/ReadVariableOpReadVariableOp-imagined_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$imagined_mean/BiasAdd/ReadVariableOp�
imagined_mean/BiasAddBiasAddimagined_mean/MatMul:product:0,imagined_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
imagined_mean/BiasAddr
IdentityIdentityimagined_mean/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������:���������:::::Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
+__inference_functional_5_layer_call_fn_2558
imaginator_input_state
imaginator_input_act
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallimaginator_input_stateimaginator_input_actunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_functional_5_layer_call_and_return_conditional_losses_25472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:���������
0
_user_specified_nameimaginator_input_state:]Y
'
_output_shapes
:���������
.
_user_specified_nameimaginator_input_act
�
�
+__inference_functional_5_layer_call_fn_2642
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_functional_5_layer_call_and_return_conditional_losses_25472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
 __inference__traced_restore_2752
file_prefix#
assignvariableop_dense_1_kernel#
assignvariableop_1_dense_1_bias+
'assignvariableop_2_imagined_mean_kernel)
%assignvariableop_3_imagined_mean_bias

identity_5��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp'assignvariableop_2_imagined_mean_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp%assignvariableop_3_imagined_mean_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4�

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
+__inference_functional_5_layer_call_fn_2528
imaginator_input_state
imaginator_input_act
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallimaginator_input_stateimaginator_input_actunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_functional_5_layer_call_and_return_conditional_losses_25172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:���������
0
_user_specified_nameimaginator_input_state:]Y
'
_output_shapes
:���������
.
_user_specified_nameimaginator_input_act
�
�
F__inference_functional_5_layer_call_and_return_conditional_losses_2497
imaginator_input_state
imaginator_input_act
dense_1_2486
dense_1_2488
imagined_mean_2491
imagined_mean_2493
identity��dense_1/StatefulPartitionedCall�%imagined_mean/StatefulPartitionedCall�
concatenate/PartitionedCallPartitionedCallimaginator_input_stateimaginator_input_act*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_24182
concatenate/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1_2486dense_1_2488*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_24382!
dense_1/StatefulPartitionedCall�
%imagined_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0imagined_mean_2491imagined_mean_2493*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_imagined_mean_layer_call_and_return_conditional_losses_24642'
%imagined_mean/StatefulPartitionedCall�
IdentityIdentity.imagined_mean/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall&^imagined_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������:���������::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2N
%imagined_mean/StatefulPartitionedCall%imagined_mean/StatefulPartitionedCall:_ [
'
_output_shapes
:���������
0
_user_specified_nameimaginator_input_state:]Y
'
_output_shapes
:���������
.
_user_specified_nameimaginator_input_act
�
�
,__inference_imagined_mean_layer_call_fn_2694

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_imagined_mean_layer_call_and_return_conditional_losses_24642
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
A__inference_dense_1_layer_call_and_return_conditional_losses_2438

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
o
E__inference_concatenate_layer_call_and_return_conditional_losses_2418

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
A__inference_dense_1_layer_call_and_return_conditional_losses_2666

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_functional_5_layer_call_and_return_conditional_losses_2517

inputs
inputs_1
dense_1_2506
dense_1_2508
imagined_mean_2511
imagined_mean_2513
identity��dense_1/StatefulPartitionedCall�%imagined_mean/StatefulPartitionedCall�
concatenate/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_24182
concatenate/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1_2506dense_1_2508*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_24382!
dense_1/StatefulPartitionedCall�
%imagined_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0imagined_mean_2511imagined_mean_2513*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_imagined_mean_layer_call_and_return_conditional_losses_24642'
%imagined_mean/StatefulPartitionedCall�
IdentityIdentity.imagined_mean/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall&^imagined_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������:���������::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2N
%imagined_mean/StatefulPartitionedCall%imagined_mean/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
U
imaginator_input_act=
&serving_default_imaginator_input_act:0���������
Y
imaginator_input_state?
(serving_default_imaginator_input_state:0���������A
imagined_mean0
StatefulPartitionedCall:0���������tensorflow/serving/predict:Ӏ
�#
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
trainable_variables
regularization_losses
	variables
		keras_api


signatures
/_default_save_signature
*0&call_and_return_all_conditional_losses
1__call__"�!
_tf_keras_network�!{"class_name": "Functional", "name": "functional_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "imaginator_input_state"}, "name": "imaginator_input_state", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "imaginator_input_act"}, "name": "imaginator_input_act", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["imaginator_input_state", 0, 0, {}], ["imaginator_input_act", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "imagined_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "imagined_mean", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["imaginator_input_state", 0, 0], ["imaginator_input_act", 0, 0]], "output_layers": [["imagined_mean", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 16]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "imaginator_input_state"}, "name": "imaginator_input_state", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "imaginator_input_act"}, "name": "imaginator_input_act", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["imaginator_input_state", 0, 0, {}], ["imaginator_input_act", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "imagined_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "imagined_mean", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["imaginator_input_state", 0, 0], ["imaginator_input_act", 0, 0]], "output_layers": [["imagined_mean", 0, 0]]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "imaginator_input_state", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "imaginator_input_state"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "imaginator_input_act", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "imaginator_input_act"}}
�
trainable_variables
regularization_losses
	variables
	keras_api
*2&call_and_return_all_conditional_losses
3__call__"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 16]}, {"class_name": "TensorShape", "items": [null, 1]}]}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*4&call_and_return_all_conditional_losses
5__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 17}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17]}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*6&call_and_return_all_conditional_losses
7__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "imagined_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "imagined_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
�
trainable_variables
regularization_losses

layers
non_trainable_variables
layer_metrics
	variables
metrics
layer_regularization_losses
1__call__
/_default_save_signature
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
,
8serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
regularization_losses

 layers
!non_trainable_variables
"layer_metrics
	variables
#metrics
$layer_regularization_losses
3__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
 : 2dense_1/kernel
: 2dense_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
trainable_variables
regularization_losses

%layers
&non_trainable_variables
'layer_metrics
	variables
(metrics
)layer_regularization_losses
5__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
&:$ 2imagined_mean/kernel
 :2imagined_mean/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
trainable_variables
regularization_losses

*layers
+non_trainable_variables
,layer_metrics
	variables
-metrics
.layer_regularization_losses
7__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
__inference__wrapped_model_2406�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *j�g
e�b
0�-
imaginator_input_state���������
.�+
imaginator_input_act���������
�2�
F__inference_functional_5_layer_call_and_return_conditional_losses_2614
F__inference_functional_5_layer_call_and_return_conditional_losses_2481
F__inference_functional_5_layer_call_and_return_conditional_losses_2594
F__inference_functional_5_layer_call_and_return_conditional_losses_2497�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_functional_5_layer_call_fn_2558
+__inference_functional_5_layer_call_fn_2528
+__inference_functional_5_layer_call_fn_2642
+__inference_functional_5_layer_call_fn_2628�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_concatenate_layer_call_and_return_conditional_losses_2649�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_concatenate_layer_call_fn_2655�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_dense_1_layer_call_and_return_conditional_losses_2666�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_dense_1_layer_call_fn_2675�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_imagined_mean_layer_call_and_return_conditional_losses_2685�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_imagined_mean_layer_call_fn_2694�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
TBR
"__inference_signature_wrapper_2574imaginator_input_actimaginator_input_state�
__inference__wrapped_model_2406�t�q
j�g
e�b
0�-
imaginator_input_state���������
.�+
imaginator_input_act���������
� "=�:
8
imagined_mean'�$
imagined_mean����������
E__inference_concatenate_layer_call_and_return_conditional_losses_2649�Z�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "%�"
�
0���������
� �
*__inference_concatenate_layer_call_fn_2655vZ�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "�����������
A__inference_dense_1_layer_call_and_return_conditional_losses_2666\/�,
%�"
 �
inputs���������
� "%�"
�
0��������� 
� y
&__inference_dense_1_layer_call_fn_2675O/�,
%�"
 �
inputs���������
� "���������� �
F__inference_functional_5_layer_call_and_return_conditional_losses_2481�|�y
r�o
e�b
0�-
imaginator_input_state���������
.�+
imaginator_input_act���������
p

 
� "%�"
�
0���������
� �
F__inference_functional_5_layer_call_and_return_conditional_losses_2497�|�y
r�o
e�b
0�-
imaginator_input_state���������
.�+
imaginator_input_act���������
p 

 
� "%�"
�
0���������
� �
F__inference_functional_5_layer_call_and_return_conditional_losses_2594�b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������
p

 
� "%�"
�
0���������
� �
F__inference_functional_5_layer_call_and_return_conditional_losses_2614�b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������
p 

 
� "%�"
�
0���������
� �
+__inference_functional_5_layer_call_fn_2528�|�y
r�o
e�b
0�-
imaginator_input_state���������
.�+
imaginator_input_act���������
p

 
� "�����������
+__inference_functional_5_layer_call_fn_2558�|�y
r�o
e�b
0�-
imaginator_input_state���������
.�+
imaginator_input_act���������
p 

 
� "�����������
+__inference_functional_5_layer_call_fn_2628�b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������
p

 
� "�����������
+__inference_functional_5_layer_call_fn_2642�b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������
p 

 
� "�����������
G__inference_imagined_mean_layer_call_and_return_conditional_losses_2685\/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� 
,__inference_imagined_mean_layer_call_fn_2694O/�,
%�"
 �
inputs��������� 
� "�����������
"__inference_signature_wrapper_2574����
� 
���
F
imaginator_input_act.�+
imaginator_input_act���������
J
imaginator_input_state0�-
imaginator_input_state���������"=�:
8
imagined_mean'�$
imagined_mean���������