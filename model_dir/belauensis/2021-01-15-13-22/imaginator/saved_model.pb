зб
ЭЃ
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
dtypetype
О
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18ЪФ
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

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
д
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB Bћ
ќ
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
#_self_saveable_object_factories

signatures
regularization_losses
		variables

trainable_variables
	keras_api
%
#_self_saveable_object_factories
%
#_self_saveable_object_factories
w
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api


kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api


kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
 	keras_api
 
 
 

0
1
2
3

0
1
2
3
­
!non_trainable_variables
"metrics
#layer_metrics
$layer_regularization_losses
regularization_losses
		variables

trainable_variables

%layers
 
 
 
 
 
 
­
&non_trainable_variables
'metrics
(layer_metrics
)layer_regularization_losses
regularization_losses
	variables
trainable_variables

*layers
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
­
+non_trainable_variables
,metrics
-layer_metrics
.layer_regularization_losses
regularization_losses
	variables
trainable_variables

/layers
`^
VARIABLE_VALUEimagined_mean/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEimagined_mean/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
­
0non_trainable_variables
1metrics
2layer_metrics
3layer_regularization_losses
regularization_losses
	variables
trainable_variables

4layers
 
 
 
 
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

$serving_default_imaginator_input_actPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

&serving_default_imaginator_input_statePlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
О
StatefulPartitionedCallStatefulPartitionedCall$serving_default_imaginator_input_act&serving_default_imaginator_input_statedense_1/kerneldense_1/biasimagined_mean/kernelimagined_mean/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *.
f)R'
%__inference_signature_wrapper_2114465
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Н
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
GPU2*0,1J 8 *)
f$R"
 __inference__traced_save_2114621
ш
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
GPU2*0,1J 8 *,
f'R%
#__inference__traced_restore_2114643Т
г
В
J__inference_imagined_mean_layer_call_and_return_conditional_losses_2114576

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :::O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
г
В
J__inference_imagined_mean_layer_call_and_return_conditional_losses_2114355

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :::O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

 
I__inference_functional_5_layer_call_and_return_conditional_losses_2114438

inputs
inputs_1
dense_1_2114427
dense_1_2114429
imagined_mean_2114432
imagined_mean_2114434
identityЂdense_1/StatefulPartitionedCallЂ%imagined_mean/StatefulPartitionedCallю
concatenate/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_21143092
concatenate/PartitionedCallЕ
dense_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1_2114427dense_1_2114429*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_21143292!
dense_1/StatefulPartitionedCallз
%imagined_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0imagined_mean_2114432imagined_mean_2114434*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *S
fNRL
J__inference_imagined_mean_layer_call_and_return_conditional_losses_21143552'
%imagined_mean/StatefulPartitionedCallЬ
IdentityIdentity.imagined_mean/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall&^imagined_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ:џџџџџџџџџ::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2N
%imagined_mean/StatefulPartitionedCall%imagined_mean/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Д
r
H__inference_concatenate_layer_call_and_return_conditional_losses_2114309

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
:џџџџџџџџџ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ
Ќ
D__inference_dense_1_layer_call_and_return_conditional_losses_2114329

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Є
Y
-__inference_concatenate_layer_call_fn_2114546
inputs_0
inputs_1
identityи
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_21143092
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
ы
М
I__inference_functional_5_layer_call_and_return_conditional_losses_2114372
imaginator_input_state
imaginator_input_act
dense_1_2114340
dense_1_2114342
imagined_mean_2114366
imagined_mean_2114368
identityЂdense_1/StatefulPartitionedCallЂ%imagined_mean/StatefulPartitionedCall
concatenate/PartitionedCallPartitionedCallimaginator_input_stateimaginator_input_act*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_21143092
concatenate/PartitionedCallЕ
dense_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1_2114340dense_1_2114342*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_21143292!
dense_1/StatefulPartitionedCallз
%imagined_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0imagined_mean_2114366imagined_mean_2114368*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *S
fNRL
J__inference_imagined_mean_layer_call_and_return_conditional_losses_21143552'
%imagined_mean/StatefulPartitionedCallЬ
IdentityIdentity.imagined_mean/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall&^imagined_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ:џџџџџџџџџ::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2N
%imagined_mean/StatefulPartitionedCall%imagined_mean/StatefulPartitionedCall:_ [
'
_output_shapes
:џџџџџџџџџ
0
_user_specified_nameimaginator_input_state:]Y
'
_output_shapes
:џџџџџџџџџ
.
_user_specified_nameimaginator_input_act
ю

/__inference_imagined_mean_layer_call_fn_2114585

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *S
fNRL
J__inference_imagined_mean_layer_call_and_return_conditional_losses_21143552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Я
Т
%__inference_signature_wrapper_2114465
imaginator_input_act
imaginator_input_state
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallimaginator_input_stateimaginator_input_actunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *+
f&R$
"__inference__wrapped_model_21142972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
'
_output_shapes
:џџџџџџџџџ
.
_user_specified_nameimaginator_input_act:_[
'
_output_shapes
:џџџџџџџџџ
0
_user_specified_nameimaginator_input_state
џ
Ы
.__inference_functional_5_layer_call_fn_2114449
imaginator_input_state
imaginator_input_act
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallimaginator_input_stateimaginator_input_actunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_functional_5_layer_call_and_return_conditional_losses_21144382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:џџџџџџџџџ
0
_user_specified_nameimaginator_input_state:]Y
'
_output_shapes
:џџџџџџџџџ
.
_user_specified_nameimaginator_input_act
Н
t
H__inference_concatenate_layer_call_and_return_conditional_losses_2114540
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
ж
Н
#__inference__traced_restore_2114643
file_prefix#
assignvariableop_dense_1_kernel#
assignvariableop_1_dense_1_bias+
'assignvariableop_2_imagined_mean_kernel)
%assignvariableop_3_imagined_mean_bias

identity_5ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slicesФ
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

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Є
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ќ
AssignVariableOp_2AssignVariableOp'assignvariableop_2_imagined_mean_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Њ
AssignVariableOp_3AssignVariableOp%assignvariableop_3_imagined_mean_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpК

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4Ќ

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
Љ
Ќ
D__inference_dense_1_layer_call_and_return_conditional_losses_2114557

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
с
~
)__inference_dense_1_layer_call_fn_2114566

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_21143292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ц
Ж
I__inference_functional_5_layer_call_and_return_conditional_losses_2114485
inputs_0
inputs_1*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource0
,imagined_mean_matmul_readvariableop_resource1
-imagined_mean_biasadd_readvariableop_resource
identityt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisЅ
concatenate/concatConcatV2inputs_0inputs_1 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2
concatenate/concatЅ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp 
dense_1/MatMulMatMulconcatenate/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_1/MatMulЄ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOpЁ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_1/ReluЗ
#imagined_mean/MatMul/ReadVariableOpReadVariableOp,imagined_mean_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#imagined_mean/MatMul/ReadVariableOpБ
imagined_mean/MatMulMatMuldense_1/Relu:activations:0+imagined_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
imagined_mean/MatMulЖ
$imagined_mean/BiasAdd/ReadVariableOpReadVariableOp-imagined_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$imagined_mean/BiasAdd/ReadVariableOpЙ
imagined_mean/BiasAddBiasAddimagined_mean/MatMul:product:0,imagined_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
imagined_mean/BiasAddr
IdentityIdentityimagined_mean/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ:џџџџџџџџџ:::::Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1

Б
 __inference__traced_save_2114621
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop3
/savev2_imagined_mean_kernel_read_readvariableop1
-savev2_imagined_mean_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_be943a3d02a04e7cacd85bfedf681e28/part2	
Const_1
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename§
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slicesђ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop/savev2_imagined_mean_kernel_read_readvariableop-savev2_imagined_mean_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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
Ц
Ж
I__inference_functional_5_layer_call_and_return_conditional_losses_2114505
inputs_0
inputs_1*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource0
,imagined_mean_matmul_readvariableop_resource1
-imagined_mean_biasadd_readvariableop_resource
identityt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisЅ
concatenate/concatConcatV2inputs_0inputs_1 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2
concatenate/concatЅ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp 
dense_1/MatMulMatMulconcatenate/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_1/MatMulЄ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOpЁ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_1/ReluЗ
#imagined_mean/MatMul/ReadVariableOpReadVariableOp,imagined_mean_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#imagined_mean/MatMul/ReadVariableOpБ
imagined_mean/MatMulMatMuldense_1/Relu:activations:0+imagined_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
imagined_mean/MatMulЖ
$imagined_mean/BiasAdd/ReadVariableOpReadVariableOp-imagined_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$imagined_mean/BiasAdd/ReadVariableOpЙ
imagined_mean/BiasAddBiasAddimagined_mean/MatMul:product:0,imagined_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
imagined_mean/BiasAddr
IdentityIdentityimagined_mean/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ:џџџџџџџџџ:::::Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
ы
М
I__inference_functional_5_layer_call_and_return_conditional_losses_2114388
imaginator_input_state
imaginator_input_act
dense_1_2114377
dense_1_2114379
imagined_mean_2114382
imagined_mean_2114384
identityЂdense_1/StatefulPartitionedCallЂ%imagined_mean/StatefulPartitionedCall
concatenate/PartitionedCallPartitionedCallimaginator_input_stateimaginator_input_act*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_21143092
concatenate/PartitionedCallЕ
dense_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1_2114377dense_1_2114379*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_21143292!
dense_1/StatefulPartitionedCallз
%imagined_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0imagined_mean_2114382imagined_mean_2114384*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *S
fNRL
J__inference_imagined_mean_layer_call_and_return_conditional_losses_21143552'
%imagined_mean/StatefulPartitionedCallЬ
IdentityIdentity.imagined_mean/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall&^imagined_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ:џџџџџџџџџ::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2N
%imagined_mean/StatefulPartitionedCall%imagined_mean/StatefulPartitionedCall:_ [
'
_output_shapes
:џџџџџџџџџ
0
_user_specified_nameimaginator_input_state:]Y
'
_output_shapes
:џџџџџџџџџ
.
_user_specified_nameimaginator_input_act

н
"__inference__wrapped_model_2114297
imaginator_input_state
imaginator_input_act7
3functional_5_dense_1_matmul_readvariableop_resource8
4functional_5_dense_1_biasadd_readvariableop_resource=
9functional_5_imagined_mean_matmul_readvariableop_resource>
:functional_5_imagined_mean_biasadd_readvariableop_resource
identity
$functional_5/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$functional_5/concatenate/concat/axisц
functional_5/concatenate/concatConcatV2imaginator_input_stateimaginator_input_act-functional_5/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2!
functional_5/concatenate/concatЬ
*functional_5/dense_1/MatMul/ReadVariableOpReadVariableOp3functional_5_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*functional_5/dense_1/MatMul/ReadVariableOpд
functional_5/dense_1/MatMulMatMul(functional_5/concatenate/concat:output:02functional_5/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
functional_5/dense_1/MatMulЫ
+functional_5/dense_1/BiasAdd/ReadVariableOpReadVariableOp4functional_5_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+functional_5/dense_1/BiasAdd/ReadVariableOpе
functional_5/dense_1/BiasAddBiasAdd%functional_5/dense_1/MatMul:product:03functional_5/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
functional_5/dense_1/BiasAdd
functional_5/dense_1/ReluRelu%functional_5/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
functional_5/dense_1/Reluо
0functional_5/imagined_mean/MatMul/ReadVariableOpReadVariableOp9functional_5_imagined_mean_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0functional_5/imagined_mean/MatMul/ReadVariableOpх
!functional_5/imagined_mean/MatMulMatMul'functional_5/dense_1/Relu:activations:08functional_5/imagined_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!functional_5/imagined_mean/MatMulн
1functional_5/imagined_mean/BiasAdd/ReadVariableOpReadVariableOp:functional_5_imagined_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1functional_5/imagined_mean/BiasAdd/ReadVariableOpэ
"functional_5/imagined_mean/BiasAddBiasAdd+functional_5/imagined_mean/MatMul:product:09functional_5/imagined_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"functional_5/imagined_mean/BiasAdd
IdentityIdentity+functional_5/imagined_mean/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ:џџџџџџџџџ:::::_ [
'
_output_shapes
:џџџџџџџџџ
0
_user_specified_nameimaginator_input_state:]Y
'
_output_shapes
:џџџџџџџџџ
.
_user_specified_nameimaginator_input_act
Б
Б
.__inference_functional_5_layer_call_fn_2114533
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_functional_5_layer_call_and_return_conditional_losses_21144382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
Б
Б
.__inference_functional_5_layer_call_fn_2114519
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_functional_5_layer_call_and_return_conditional_losses_21144082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
џ
Ы
.__inference_functional_5_layer_call_fn_2114419
imaginator_input_state
imaginator_input_act
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallimaginator_input_stateimaginator_input_actunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_functional_5_layer_call_and_return_conditional_losses_21144082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:џџџџџџџџџ
0
_user_specified_nameimaginator_input_state:]Y
'
_output_shapes
:џџџџџџџџџ
.
_user_specified_nameimaginator_input_act

 
I__inference_functional_5_layer_call_and_return_conditional_losses_2114408

inputs
inputs_1
dense_1_2114397
dense_1_2114399
imagined_mean_2114402
imagined_mean_2114404
identityЂdense_1/StatefulPartitionedCallЂ%imagined_mean/StatefulPartitionedCallю
concatenate/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_21143092
concatenate/PartitionedCallЕ
dense_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1_2114397dense_1_2114399*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_21143292!
dense_1/StatefulPartitionedCallз
%imagined_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0imagined_mean_2114402imagined_mean_2114404*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *S
fNRL
J__inference_imagined_mean_layer_call_and_return_conditional_losses_21143552'
%imagined_mean/StatefulPartitionedCallЬ
IdentityIdentity.imagined_mean/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall&^imagined_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ:џџџџџџџџџ::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2N
%imagined_mean/StatefulPartitionedCall%imagined_mean/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ѕ
serving_default
U
imaginator_input_act=
&serving_default_imaginator_input_act:0џџџџџџџџџ
Y
imaginator_input_state?
(serving_default_imaginator_input_state:0џџџџџџџџџA
imagined_mean0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:о
$
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
#_self_saveable_object_factories

signatures
regularization_losses
		variables

trainable_variables
	keras_api
*5&call_and_return_all_conditional_losses
6_default_save_signature
7__call__"И!
_tf_keras_network!{"class_name": "Functional", "name": "functional_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "imaginator_input_state"}, "name": "imaginator_input_state", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "imaginator_input_act"}, "name": "imaginator_input_act", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["imaginator_input_state", 0, 0, {}], ["imaginator_input_act", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "imagined_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "imagined_mean", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["imaginator_input_state", 0, 0], ["imaginator_input_act", 0, 0]], "output_layers": [["imagined_mean", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 16]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "imaginator_input_state"}, "name": "imaginator_input_state", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "imaginator_input_act"}, "name": "imaginator_input_act", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["imaginator_input_state", 0, 0, {}], ["imaginator_input_act", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "imagined_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "imagined_mean", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["imaginator_input_state", 0, 0], ["imaginator_input_act", 0, 0]], "output_layers": [["imagined_mean", 0, 0]]}}}
Ў
#_self_saveable_object_factories"
_tf_keras_input_layerц{"class_name": "InputLayer", "name": "imaginator_input_state", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "imaginator_input_state"}}
Ј
#_self_saveable_object_factories"
_tf_keras_input_layerр{"class_name": "InputLayer", "name": "imaginator_input_act", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "imaginator_input_act"}}
э
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
*8&call_and_return_all_conditional_losses
9__call__"Й
_tf_keras_layer{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 16]}, {"class_name": "TensorShape", "items": [null, 1]}]}


kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
*:&call_and_return_all_conditional_losses
;__call__"Ы
_tf_keras_layerБ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 17}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17]}}
Ѓ

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
 	keras_api
*<&call_and_return_all_conditional_losses
=__call__"й
_tf_keras_layerП{"class_name": "Dense", "name": "imagined_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "imagined_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
 "
trackable_dict_wrapper
,
>serving_default"
signature_map
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
Ъ
!non_trainable_variables
"metrics
#layer_metrics
$layer_regularization_losses
regularization_losses
		variables

trainable_variables

%layers
7__call__
6_default_save_signature
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
&non_trainable_variables
'metrics
(layer_metrics
)layer_regularization_losses
regularization_losses
	variables
trainable_variables

*layers
9__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
 : 2dense_1/kernel
: 2dense_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
+non_trainable_variables
,metrics
-layer_metrics
.layer_regularization_losses
regularization_losses
	variables
trainable_variables

/layers
;__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
&:$ 2imagined_mean/kernel
 :2imagined_mean/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
0non_trainable_variables
1metrics
2layer_metrics
3layer_regularization_losses
regularization_losses
	variables
trainable_variables

4layers
=__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
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
ђ2я
I__inference_functional_5_layer_call_and_return_conditional_losses_2114372
I__inference_functional_5_layer_call_and_return_conditional_losses_2114485
I__inference_functional_5_layer_call_and_return_conditional_losses_2114505
I__inference_functional_5_layer_call_and_return_conditional_losses_2114388Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Є2Ё
"__inference__wrapped_model_2114297њ
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *jЂg
eb
0-
imaginator_input_stateџџџџџџџџџ
.+
imaginator_input_actџџџџџџџџџ
2
.__inference_functional_5_layer_call_fn_2114533
.__inference_functional_5_layer_call_fn_2114449
.__inference_functional_5_layer_call_fn_2114419
.__inference_functional_5_layer_call_fn_2114519Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ђ2я
H__inference_concatenate_layer_call_and_return_conditional_losses_2114540Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_concatenate_layer_call_fn_2114546Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_dense_1_layer_call_and_return_conditional_losses_2114557Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_dense_1_layer_call_fn_2114566Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
є2ё
J__inference_imagined_mean_layer_call_and_return_conditional_losses_2114576Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
й2ж
/__inference_imagined_mean_layer_call_fn_2114585Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
WBU
%__inference_signature_wrapper_2114465imaginator_input_actimaginator_input_stateт
"__inference__wrapped_model_2114297ЛtЂq
jЂg
eb
0-
imaginator_input_stateџџџџџџџџџ
.+
imaginator_input_actџџџџџџџџџ
Њ "=Њ:
8
imagined_mean'$
imagined_meanџџџџџџџџџа
H__inference_concatenate_layer_call_and_return_conditional_losses_2114540ZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 Ї
-__inference_concatenate_layer_call_fn_2114546vZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
Њ "џџџџџџџџџЄ
D__inference_dense_1_layer_call_and_return_conditional_losses_2114557\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ 
 |
)__inference_dense_1_layer_call_fn_2114566O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ љ
I__inference_functional_5_layer_call_and_return_conditional_losses_2114372Ћ|Ђy
rЂo
eb
0-
imaginator_input_stateџџџџџџџџџ
.+
imaginator_input_actџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 љ
I__inference_functional_5_layer_call_and_return_conditional_losses_2114388Ћ|Ђy
rЂo
eb
0-
imaginator_input_stateџџџџџџџџџ
.+
imaginator_input_actџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 п
I__inference_functional_5_layer_call_and_return_conditional_losses_2114485bЂ_
XЂU
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 п
I__inference_functional_5_layer_call_and_return_conditional_losses_2114505bЂ_
XЂU
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 б
.__inference_functional_5_layer_call_fn_2114419|Ђy
rЂo
eb
0-
imaginator_input_stateџџџџџџџџџ
.+
imaginator_input_actџџџџџџџџџ
p

 
Њ "џџџџџџџџџб
.__inference_functional_5_layer_call_fn_2114449|Ђy
rЂo
eb
0-
imaginator_input_stateџџџџџџџџџ
.+
imaginator_input_actџџџџџџџџџ
p 

 
Њ "џџџџџџџџџЗ
.__inference_functional_5_layer_call_fn_2114519bЂ_
XЂU
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
p

 
Њ "џџџџџџџџџЗ
.__inference_functional_5_layer_call_fn_2114533bЂ_
XЂU
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
p 

 
Њ "џџџџџџџџџЊ
J__inference_imagined_mean_layer_call_and_return_conditional_losses_2114576\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 
/__inference_imagined_mean_layer_call_fn_2114585O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ
%__inference_signature_wrapper_2114465ьЄЂ 
Ђ 
Њ
F
imaginator_input_act.+
imaginator_input_actџџџџџџџџџ
J
imaginator_input_state0-
imaginator_input_stateџџџџџџџџџ"=Њ:
8
imagined_mean'$
imagined_meanџџџџџџџџџ