var e=Object.create,t=Object.defineProperty,n=Object.getOwnPropertyDescriptor,r=Object.getOwnPropertyNames,i=Object.getPrototypeOf,a=Object.prototype.hasOwnProperty,o=(e,t)=>()=>(t||(e((t={exports:{}}).exports,t),e=null),t.exports),s=(e,n)=>{let r={};for(var i in e)t(r,i,{get:e[i],enumerable:!0});return n||t(r,Symbol.toStringTag,{value:`Module`}),r},c=(e,i,o,s)=>{if(i&&typeof i==`object`||typeof i==`function`)for(var c=r(i),l=0,u=c.length,d;l<u;l++)d=c[l],!a.call(e,d)&&d!==o&&t(e,d,{get:(e=>i[e]).bind(null,d),enumerable:!(s=n(i,d))||s.enumerable});return e},l=(n,r,a)=>(a=n==null?{}:e(i(n)),c(r||!n||!n.__esModule?t(a,`default`,{value:n,enumerable:!0}):a,n)),u=1e-7,d=1e-4,f=class{constructor(e,t){this.backend=e,this.dataMover=t,this.data=new WeakMap,this.dataIdsCount=0}get(e){return this.data.has(e)||this.dataMover.moveData(this.backend,e),this.data.get(e)}set(e,t){this.dataIdsCount++,this.data.set(e,t)}has(e){return this.data.has(e)}delete(e){return this.dataIdsCount--,this.data.delete(e)}numDataIds(){return this.dataIdsCount}},p=class{refCount(e){return m(`refCount`)}incRef(e){return m(`incRef`)}timerAvailable(){return!0}time(e){return m(`time`)}read(e){return m(`read`)}readSync(e){return m(`readSync`)}readToGPU(e,t){return m(`readToGPU`)}numDataIds(){return m(`numDataIds`)}disposeData(e,t){return m(`disposeData`)}write(e,t,n){return m(`write`)}move(e,t,n,r,i){return m(`move`)}createTensorFromGPUData(e,t,n){return m(`createTensorFromGPUData`)}memory(){return m(`memory`)}floatPrecision(){return m(`floatPrecision`)}epsilon(){return this.floatPrecision()===32?u:d}dispose(){return m(`dispose`)}};function m(e){throw Error(`'${e}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}function h(e,t,n){return Math.max(e,Math.min(t,n))}function g(e){return e%2==0?e:e+1}function _(e,t,n){let r=e[t];e[t]=e[n],e[n]=r}function v(e){let t=0;for(let n=0;n<e.length;n++)t+=e[n];return t}function y(e,t){if(!e)throw Error(typeof t==`string`?t:t())}function b(e,t,n=``){y(w(e,t),()=>n+` Shapes ${e} and ${t} must match`)}function x(e){y(e!=null,()=>`The input to the tensor constructor must be a non-null value.`)}function S(e){if(e.length===0)return 1;let t=e[0];for(let n=1;n<e.length;n++)t*=e[n];return t}function C(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==null&&t[n]!==null&&e[n]!==t[n])return!1;return!0}function w(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==t[n])return!1;return!0}function ee(e){return e%1==0}function T(e){let t=Math.ceil(Math.sqrt(e));return[t,Math.ceil(e/t)]}function E(e,t){return t<=e.length?e:e+` `.repeat(t-e.length)}function te(e,t=e=>0,n,r){return new Promise((i,a)=>{let o=0,s=()=>{if(e()){i();return}o++;let c=t(o);if(n!=null&&o>=n){a();return}r==null?setTimeout(s,c):r(s,c)};s()})}function ne(e,t){let n=1,r=-1;for(let t=0;t<e.length;++t)if(e[t]>=0)n*=e[t];else if(e[t]===-1){if(r!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${r} and dim ${t}`);r=t}else if(e[t]<0)throw Error(`Shapes can not be < 0. Found ${e[t]} at dim ${t}`);if(r===-1){if(t>0&&t!==n)throw Error(`Size(${t}) must match the product of shape ${e}`);return e}if(n===0)throw Error(`Cannot infer the missing size in [${e}] when there are 0 elements`);if(t%n!==0)throw Error(`The implicit shape can't be a fractional number. Got ${t} / ${n}`);let i=e.slice();return i[r]=t/n,i}function D(e,t){let n=t.length;return e=e==null?t.map((e,t)=>t):[].concat(e),y(e.every(e=>e>=-n&&e<n),()=>`All values in axis param must be in range [-${n}, ${n}) but got axis ${e}`),y(e.every(e=>ee(e)),()=>`All values in axis param must be integers but got axis ${e}`),e.map(e=>e<0?n+e:e)}function re(e,t){let n=[],r=[],i=t!=null&&Array.isArray(t)&&t.length===0,a=t==null||i?null:D(t,e).sort(),o=0;for(let t=0;t<e.length;++t){if(a!=null){if(a[o]===t&&e[t]!==1)throw Error(`Can't squeeze axis ${t} since its dim '${e[t]}' is not 1`);(a[o]==null||a[o]>t)&&e[t]===1&&(n.push(e[t]),r.push(t)),a[o]<=t&&o++}e[t]!==1&&(n.push(e[t]),r.push(t))}return{newShape:n,keptDims:r}}function ie(e,t){return ae(e,t)}function ae(e,t){let n=null;if(e==null||e===`float32`)n=new Float32Array(t);else if(e===`int32`)n=new Int32Array(t);else if(e===`bool`)n=new Uint8Array(t);else if(e===`string`)n=Array(t);else throw Error(`Unknown data type ${e}`);return n}function oe(e,t){for(let n=0;n<e.length;n++){let r=e[n];if(isNaN(r)||!isFinite(r))throw Error(`A tensor of type ${t} being uploaded contains ${r}.`)}}function se(e){return e===`bool`||e===`complex64`||e===`float32`||e===`int32`||e===`string`}function ce(e,t){return!(t===`complex64`||t===`float32`&&e!==`complex64`||t===`int32`&&e!==`float32`&&e!==`complex64`||t===`bool`&&e===`bool`)}function le(e){if(e===`float32`||e===`int32`)return 4;if(e===`complex64`)return 8;if(e===`bool`)return 1;throw Error(`Unknown dtype ${e}`)}function ue(e){if(e==null)return 0;let t=0;return e.forEach(e=>t+=e.length),t}function de(e){return typeof e==`string`||e instanceof String}function fe(e){return typeof e==`boolean`}function pe(e){return typeof e==`number`}function me(e){return Array.isArray(e)?me(e[0]):e instanceof Float32Array?`float32`:e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray?`int32`:pe(e)?`float32`:de(e)?`string`:fe(e)?`bool`:`float32`}function he(e){return!!(e&&e.constructor&&e.call&&e.apply)}function ge(e,t){for(let n=t;n<e;++n)if(e%n===0)return n;return e}function _e(e){let t=e.length;if(t<2)return[];let n=Array(t-1);n[t-2]=e[t-1];for(let r=t-3;r>=0;--r)n[r]=n[r+1]*e[r+1];return n}function ve(e,t,n,r=!1){let i=[];if(t.length===1){let a=t[0]*(r?2:1);for(let t=0;t<a;t++)i[t]=n[e+t]}else{let a=t[0],o=t.slice(1),s=o.reduce((e,t)=>e*t)*(r?2:1);for(let t=0;t<a;t++)i[t]=ve(e+t*s,o,n,r)}return i}function ye(e,t,n=!1){if(e.length===0)return t[0];let r=e.reduce((e,t)=>e*t)*(n?2:1);if(r===0)return[];if(r!==t.length)throw Error(`[${e}] does not match the input size ${t.length}${n?` for a complex tensor`:``}.`);return ve(0,e,t,n)}function be(e,t){let n=xe(e,t);for(let e=0;e<n.length;e++)n[e]=1;return n}function xe(e,t){if(t==null||t===`float32`||t===`complex64`)return new Float32Array(e);if(t===`int32`)return new Int32Array(e);if(t===`bool`)return new Uint8Array(e);throw Error(`Unknown data type ${t}`)}function Se(e){e.forEach(t=>{y(Number.isInteger(t)&&t>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${e}].`)})}function Ce(e,t,n){if(t===0)return 0;if(t===1)return e[0];let r=e[e.length-1];for(let t=0;t<e.length-1;++t)r+=n[t]*e[t];return r}function we(e,t,n){if(t===0)return[];if(t===1)return[e];let r=Array(t);for(let t=0;t<r.length-1;++t)r[t]=Math.floor(e/n[t]),e-=r[t]*n[t];return r[r.length-1]=e,r}function Te(e){return e&&e.then&&typeof e.then==`function`}var Ee=`tfjsflags`,De=class{constructor(e){this.global=e,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=Oe,this.populateURLFlags()}setPlatform(e,t){this.platform!=null&&(O().getBool(`IS_TEST`)||O().getBool(`PROD`)||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${e}.`)),this.platformName=e,this.platform=t}registerFlag(e,t,n){if(this.flagRegistry[e]={evaluationFn:t,setHook:n},this.urlFlags[e]!=null){let t=this.urlFlags[e];O().getBool(`IS_TEST`)||O().getBool(`PROD`)||console.warn(`Setting feature override from URL ${e}: ${t}.`),this.set(e,t)}}async getAsync(e){return e in this.flags||(this.flags[e]=await this.evaluateFlag(e)),this.flags[e]}get(e){if(e in this.flags)return this.flags[e];let t=this.evaluateFlag(e);if(Te(t))throw Error(`Flag ${e} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[e]=t,this.flags[e]}getNumber(e){return this.get(e)}getBool(e){return this.get(e)}getString(e){return this.get(e)}getFlags(){return this.flags}get features(){return this.flags}set(e,t){if(this.flagRegistry[e]==null)throw Error(`Cannot set flag ${e} as it has not been registered.`);this.flags[e]=t,this.flagRegistry[e].setHook!=null&&this.flagRegistry[e].setHook(t)}evaluateFlag(e){if(this.flagRegistry[e]==null)throw Error(`Cannot evaluate flag '${e}': no evaluation function found.`);return this.flagRegistry[e].evaluationFn()}setFlags(e){this.flags=Object.assign({},e)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(this.global===void 0||this.global.location===void 0||this.global.location.search===void 0)return;let e=this.getQueryParams(this.global.location.search);Ee in e&&e[Ee].split(`,`).forEach(e=>{let[t,n]=e.split(`:`);this.urlFlags[t]=Ae(t,n)})}};function Oe(e){let t={};return e.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(e,...n)=>(ke(t,n[0],n[1]),n.join(`=`))),t}function ke(e,t,n){e[decodeURIComponent(t)]=decodeURIComponent(n||``)}function Ae(e,t){let n=t.toLowerCase();return n===`true`||n===`false`?n===`true`:`${+n}`===n?+n:t}function O(){return je}var je=null;function Me(e){je=e}var Ne;function Pe(){if(Ne==null){let e;if(typeof window<`u`)e=window;else if(typeof global<`u`)e=global;else if(typeof process<`u`)e=process;else if(typeof self<`u`)e=self;else throw Error(`Could not find a global object`);Ne=e}return Ne}function Fe(){let e=Pe();return e._tfGlobals??=new Map,e._tfGlobals}function Ie(e,t){let n=Fe();if(n.has(e))return n.get(e);{let r=t();return n.set(e,r),n.get(e)}}var Le=`Acos`,Re=`Acosh`,ze=`AddN`,Be=`ArgMax`,Ve=`ArgMin`,He=`Asin`,Ue=`Asinh`,We=`Atan`,Ge=`Atanh`,Ke=`Atan2`,qe=`AvgPool`,Je=`AvgPoolGrad`,Ye=`AvgPool3D`,Xe=`AvgPool3DGrad`,Ze=`BatchMatMul`,Qe=`BatchToSpaceND`,$e=`Bincount`,et=`BitwiseAnd`,tt=`BroadcastArgs`,nt=`Cast`,rt=`Ceil`,it=`ClipByValue`,at=`Complex`,ot=`ComplexAbs`,st=`Concat`,ct=`Conv2D`,lt=`Conv2DBackpropFilter`,ut=`Conv2DBackpropInput`,dt=`Conv3D`,ft=`Conv3DBackpropFilterV2`,pt=`Conv3DBackpropInputV2`,mt=`Cosh`,ht=`Cumprod`,gt=`Cumsum`,_t=`CropAndResize`,vt=`DenseBincount`,yt=`DepthToSpace`,bt=`DepthwiseConv2dNative`,xt=`DepthwiseConv2dNativeBackpropFilter`,St=`DepthwiseConv2dNativeBackpropInput`,Ct=`Diag`,wt=`Dilation2D`,Tt=`RealDiv`,Et=`Einsum`,Dt=`EluGrad`,Ot=`Equal`,kt=`ExpandDims`,At=`Expm1`,jt=`Fill`,Mt=`FlipLeftRight`,Nt=`Floor`,Pt=`FloorDiv`,Ft=`FusedBatchNorm`,It=`GatherV2`,Lt=`GatherNd`,Rt=`Greater`,zt=`GreaterEqual`,Bt=`Identity`,Vt=`IFFT`,Ht=`Imag`,Ut=`IsFinite`,Wt=`IsInf`,Gt=`IsNan`,Kt=`LeakyRelu`,qt=`Less`,Jt=`LessEqual`,Yt=`LinSpace`,Xt=`Log1p`,Zt=`LogicalAnd`,Qt=`LogicalNot`,$t=`LogicalOr`,en=`LRNGrad`,tn=`Maximum`,nn=`MaxPool`,rn=`MaxPoolGrad`,an=`MaxPool3D`,on=`MaxPool3DGrad`,sn=`MaxPoolWithArgmax`,cn=`Mean`,ln=`Minimum`,un=`MirrorPad`,dn=`Multinomial`,fn=`Multiply`,pn=`NotEqual`,mn=`NonMaxSuppressionV3`,hn=`NonMaxSuppressionV4`,gn=`NonMaxSuppressionV5`,_n=`OnesLike`,vn=`OneHot`,yn=`Pack`,bn=`PadV2`,xn=`Prelu`,Sn=`Prod`,Cn=`RaggedGather`,wn=`RaggedRange`,Tn=`RaggedTensorToTensor`,En=`Range`,Dn=`Real`,On=`Reciprocal`,kn=`Relu`,An=`Reshape`,jn=`ResizeNearestNeighbor`,Mn=`ResizeNearestNeighborGrad`,Nn=`ResizeBilinear`,Pn=`ResizeBilinearGrad`,Fn=`Relu6`,In=`Reverse`,Ln=`Round`,Rn=`Rsqrt`,zn=`ScatterNd`,Bn=`TensorScatterUpdate`,Vn=`SearchSorted`,Hn=`Select`,Un=`Selu`,Wn=`Slice`,Gn=`Sinh`,Kn=`Sign`,qn=`Sigmoid`,Jn=`Softplus`,Yn=`Sqrt`,Xn=`SpaceToBatchND`,Zn=`SplitV`,Qn=`Softmax`,$n=`SparseFillEmptyRows`,er=`SparseReshape`,tr=`SparseSegmentMean`,nr=`SparseSegmentSum`,rr=`SparseToDense`,ir=`SquaredDifference`,ar=`Square`,or=`StaticRegexReplace`,sr=`StridedSlice`,cr=`StringNGrams`,lr=`StringSplit`,ur=`StringToHashBucketFast`,dr=`Tanh`,fr=`Tile`,pr=`TopK`,mr=`Transform`,hr=`Transpose`,gr=`Unique`,_r=`Unpack`,vr=`UnsortedSegmentSum`,yr=`ZerosLike`,br=`Step`,xr=`FromPixels`,Sr=`RotateWithOffset`,Cr=`_FusedMatMul`,wr=`FusedConv2D`,Tr=`FusedDepthwiseConv2D`;function Er(...e){O().getBool(`IS_TEST`)||O().getBool(`PROD`)||console.warn(...e)}function Dr(...e){O().getBool(`IS_TEST`)||O().getBool(`PROD`)||console.log(...e)}var Or=Ie(`kernelRegistry`,()=>new Map),kr=Ie(`gradRegistry`,()=>new Map);function Ar(e,t){let n=Pr(e,t);return Or.get(n)}function jr(e){return kr.get(e)}function Mr(e){let t=Or.entries(),n=[];for(;;){let{done:r,value:i}=t.next();if(r)break;let[a,o]=i,[s]=a.split(`_`);s===e&&n.push(o)}return n}function Nr(e){let{kernelName:t,backendName:n}=e,r=Pr(t,n);Or.has(r)&&Er(`The kernel '${t}' for backend '${n}' is already registered`),Or.set(r,e)}function Pr(e,t){return`${t}_${e}`}function Fr(e){return e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray}var Ir=l(o(((e,t)=>{t.exports=r;var n=null;try{n=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch{}function r(e,t,n){this.low=e|0,this.high=t|0,this.unsigned=!!n}r.prototype.__isLong__,Object.defineProperty(r.prototype,`__isLong__`,{value:!0});function i(e){return(e&&e.__isLong__)===!0}r.isLong=i;var a={},o={};function s(e,t){var n,r,i;return t?(e>>>=0,(i=0<=e&&e<256)&&(r=o[e],r)?r:(n=l(e,(e|0)<0?-1:0,!0),i&&(o[e]=n),n)):(e|=0,(i=-128<=e&&e<128)&&(r=a[e],r)?r:(n=l(e,e<0?-1:0,!1),i&&(a[e]=n),n))}r.fromInt=s;function c(e,t){if(isNaN(e))return t?b:y;if(t){if(e<0)return b;if(e>=g)return ee}else{if(e<=-_)return T;if(e+1>=_)return w}return e<0?c(-e,t).neg():l(e%h|0,e/h|0,t)}r.fromNumber=c;function l(e,t,n){return new r(e,t,n)}r.fromBits=l;var u=Math.pow;function d(e,t,n){if(e.length===0)throw Error(`empty string`);if(e===`NaN`||e===`Infinity`||e===`+Infinity`||e===`-Infinity`)return y;if(typeof t==`number`?(n=t,t=!1):t=!!t,n||=10,n<2||36<n)throw RangeError(`radix`);var r;if((r=e.indexOf(`-`))>0)throw Error(`interior hyphen`);if(r===0)return d(e.substring(1),t,n).neg();for(var i=c(u(n,8)),a=y,o=0;o<e.length;o+=8){var s=Math.min(8,e.length-o),l=parseInt(e.substring(o,o+s),n);if(s<8){var f=c(u(n,s));a=a.mul(f).add(c(l))}else a=a.mul(i),a=a.add(c(l))}return a.unsigned=t,a}r.fromString=d;function f(e,t){return typeof e==`number`?c(e,t):typeof e==`string`?d(e,t):l(e.low,e.high,typeof t==`boolean`?t:e.unsigned)}r.fromValue=f;var p=65536,m=1<<24,h=p*p,g=h*h,_=g/2,v=s(m),y=s(0);r.ZERO=y;var b=s(0,!0);r.UZERO=b;var x=s(1);r.ONE=x;var S=s(1,!0);r.UONE=S;var C=s(-1);r.NEG_ONE=C;var w=l(-1,2147483647,!1);r.MAX_VALUE=w;var ee=l(-1,-1,!0);r.MAX_UNSIGNED_VALUE=ee;var T=l(0,-2147483648,!1);r.MIN_VALUE=T;var E=r.prototype;E.toInt=function(){return this.unsigned?this.low>>>0:this.low},E.toNumber=function(){return this.unsigned?(this.high>>>0)*h+(this.low>>>0):this.high*h+(this.low>>>0)},E.toString=function(e){if(e||=10,e<2||36<e)throw RangeError(`radix`);if(this.isZero())return`0`;if(this.isNegative())if(this.eq(T)){var t=c(e),n=this.div(t),r=n.mul(t).sub(this);return n.toString(e)+r.toInt().toString(e)}else return`-`+this.neg().toString(e);for(var i=c(u(e,6),this.unsigned),a=this,o=``;;){var s=a.div(i),l=(a.sub(s.mul(i)).toInt()>>>0).toString(e);if(a=s,a.isZero())return l+o;for(;l.length<6;)l=`0`+l;o=``+l+o}},E.getHighBits=function(){return this.high},E.getHighBitsUnsigned=function(){return this.high>>>0},E.getLowBits=function(){return this.low},E.getLowBitsUnsigned=function(){return this.low>>>0},E.getNumBitsAbs=function(){if(this.isNegative())return this.eq(T)?64:this.neg().getNumBitsAbs();for(var e=this.high==0?this.low:this.high,t=31;t>0&&!(e&1<<t);t--);return this.high==0?t+1:t+33},E.isZero=function(){return this.high===0&&this.low===0},E.eqz=E.isZero,E.isNegative=function(){return!this.unsigned&&this.high<0},E.isPositive=function(){return this.unsigned||this.high>=0},E.isOdd=function(){return(this.low&1)==1},E.isEven=function(){return(this.low&1)==0},E.equals=function(e){return i(e)||(e=f(e)),this.unsigned!==e.unsigned&&this.high>>>31==1&&e.high>>>31==1?!1:this.high===e.high&&this.low===e.low},E.eq=E.equals,E.notEquals=function(e){return!this.eq(e)},E.neq=E.notEquals,E.ne=E.notEquals,E.lessThan=function(e){return this.comp(e)<0},E.lt=E.lessThan,E.lessThanOrEqual=function(e){return this.comp(e)<=0},E.lte=E.lessThanOrEqual,E.le=E.lessThanOrEqual,E.greaterThan=function(e){return this.comp(e)>0},E.gt=E.greaterThan,E.greaterThanOrEqual=function(e){return this.comp(e)>=0},E.gte=E.greaterThanOrEqual,E.ge=E.greaterThanOrEqual,E.compare=function(e){if(i(e)||(e=f(e)),this.eq(e))return 0;var t=this.isNegative(),n=e.isNegative();return t&&!n?-1:!t&&n?1:this.unsigned?e.high>>>0>this.high>>>0||e.high===this.high&&e.low>>>0>this.low>>>0?-1:1:this.sub(e).isNegative()?-1:1},E.comp=E.compare,E.negate=function(){return!this.unsigned&&this.eq(T)?T:this.not().add(x)},E.neg=E.negate,E.add=function(e){i(e)||(e=f(e));var t=this.high>>>16,n=this.high&65535,r=this.low>>>16,a=this.low&65535,o=e.high>>>16,s=e.high&65535,c=e.low>>>16,u=e.low&65535,d=0,p=0,m=0,h=0;return h+=a+u,m+=h>>>16,h&=65535,m+=r+c,p+=m>>>16,m&=65535,p+=n+s,d+=p>>>16,p&=65535,d+=t+o,d&=65535,l(m<<16|h,d<<16|p,this.unsigned)},E.subtract=function(e){return i(e)||(e=f(e)),this.add(e.neg())},E.sub=E.subtract,E.multiply=function(e){if(this.isZero())return y;if(i(e)||(e=f(e)),n)return l(n.mul(this.low,this.high,e.low,e.high),n.get_high(),this.unsigned);if(e.isZero())return y;if(this.eq(T))return e.isOdd()?T:y;if(e.eq(T))return this.isOdd()?T:y;if(this.isNegative())return e.isNegative()?this.neg().mul(e.neg()):this.neg().mul(e).neg();if(e.isNegative())return this.mul(e.neg()).neg();if(this.lt(v)&&e.lt(v))return c(this.toNumber()*e.toNumber(),this.unsigned);var t=this.high>>>16,r=this.high&65535,a=this.low>>>16,o=this.low&65535,s=e.high>>>16,u=e.high&65535,d=e.low>>>16,p=e.low&65535,m=0,h=0,g=0,_=0;return _+=o*p,g+=_>>>16,_&=65535,g+=a*p,h+=g>>>16,g&=65535,g+=o*d,h+=g>>>16,g&=65535,h+=r*p,m+=h>>>16,h&=65535,h+=a*d,m+=h>>>16,h&=65535,h+=o*u,m+=h>>>16,h&=65535,m+=t*p+r*d+a*u+o*s,m&=65535,l(g<<16|_,m<<16|h,this.unsigned)},E.mul=E.multiply,E.divide=function(e){if(i(e)||(e=f(e)),e.isZero())throw Error(`division by zero`);if(n)return!this.unsigned&&this.high===-2147483648&&e.low===-1&&e.high===-1?this:l((this.unsigned?n.div_u:n.div_s)(this.low,this.high,e.low,e.high),n.get_high(),this.unsigned);if(this.isZero())return this.unsigned?b:y;var t,r,a;if(this.unsigned){if(e.unsigned||(e=e.toUnsigned()),e.gt(this))return b;if(e.gt(this.shru(1)))return S;a=b}else{if(this.eq(T))return e.eq(x)||e.eq(C)?T:e.eq(T)?x:(t=this.shr(1).div(e).shl(1),t.eq(y)?e.isNegative()?x:C:(r=this.sub(e.mul(t)),a=t.add(r.div(e)),a));if(e.eq(T))return this.unsigned?b:y;if(this.isNegative())return e.isNegative()?this.neg().div(e.neg()):this.neg().div(e).neg();if(e.isNegative())return this.div(e.neg()).neg();a=y}for(r=this;r.gte(e);){t=Math.max(1,Math.floor(r.toNumber()/e.toNumber()));for(var o=Math.ceil(Math.log(t)/Math.LN2),s=o<=48?1:u(2,o-48),d=c(t),p=d.mul(e);p.isNegative()||p.gt(r);)t-=s,d=c(t,this.unsigned),p=d.mul(e);d.isZero()&&(d=x),a=a.add(d),r=r.sub(p)}return a},E.div=E.divide,E.modulo=function(e){return i(e)||(e=f(e)),n?l((this.unsigned?n.rem_u:n.rem_s)(this.low,this.high,e.low,e.high),n.get_high(),this.unsigned):this.sub(this.div(e).mul(e))},E.mod=E.modulo,E.rem=E.modulo,E.not=function(){return l(~this.low,~this.high,this.unsigned)},E.and=function(e){return i(e)||(e=f(e)),l(this.low&e.low,this.high&e.high,this.unsigned)},E.or=function(e){return i(e)||(e=f(e)),l(this.low|e.low,this.high|e.high,this.unsigned)},E.xor=function(e){return i(e)||(e=f(e)),l(this.low^e.low,this.high^e.high,this.unsigned)},E.shiftLeft=function(e){return i(e)&&(e=e.toInt()),(e&=63)==0?this:e<32?l(this.low<<e,this.high<<e|this.low>>>32-e,this.unsigned):l(0,this.low<<e-32,this.unsigned)},E.shl=E.shiftLeft,E.shiftRight=function(e){return i(e)&&(e=e.toInt()),(e&=63)==0?this:e<32?l(this.low>>>e|this.high<<32-e,this.high>>e,this.unsigned):l(this.high>>e-32,this.high>=0?0:-1,this.unsigned)},E.shr=E.shiftRight,E.shiftRightUnsigned=function(e){if(i(e)&&(e=e.toInt()),e&=63,e===0)return this;var t=this.high;if(e<32){var n=this.low;return l(n>>>e|t<<32-e,t>>>e,this.unsigned)}else if(e===32)return l(t,0,this.unsigned);else return l(t>>>e-32,0,this.unsigned)},E.shru=E.shiftRightUnsigned,E.shr_u=E.shiftRightUnsigned,E.toSigned=function(){return this.unsigned?l(this.low,this.high,!1):this},E.toUnsigned=function(){return this.unsigned?this:l(this.low,this.high,!0)},E.toBytes=function(e){return e?this.toBytesLE():this.toBytesBE()},E.toBytesLE=function(){var e=this.high,t=this.low;return[t&255,t>>>8&255,t>>>16&255,t>>>24,e&255,e>>>8&255,e>>>16&255,e>>>24]},E.toBytesBE=function(){var e=this.high,t=this.low;return[e>>>24,e>>>16&255,e>>>8&255,e&255,t>>>24,t>>>16&255,t>>>8&255,t&255]},r.fromBytes=function(e,t,n){return n?r.fromBytesLE(e,t):r.fromBytesBE(e,t)},r.fromBytesLE=function(e,t){return new r(e[0]|e[1]<<8|e[2]<<16|e[3]<<24,e[4]|e[5]<<8|e[6]<<16|e[7]<<24,t)},r.fromBytesBE=function(e,t){return new r(e[4]<<24|e[5]<<16|e[6]<<8|e[7],e[0]<<24|e[1]<<16|e[2]<<8|e[3],t)}}))()),Lr=Ir.default||Ir;function Rr(e){return Lr.fromString(e,!0,16)}var zr=Rr(`c3a5c85c97cb3127`),Br=Rr(`b492b66fbe98f273`),Vr=Rr(`9ae16a3b2f90404f`);function Hr(e){return e.xor(e.shru(47))}function Ur(e,t,n){let r=e.slice(t,t+n);return Lr.fromBytes(Array.from(r),!0,!0)}function k(e,t){return Ur(e,t,8)}function Wr(e,t){return Ur(e,t,4)}function Gr(e,t){return t===0?e:e.shru(t).or(e.shl(64-t))}function Kr(e,t,n=Rr(`9ddfea08eb382d69`)){let r=e.xor(t).mul(n);r=r.xor(r.shru(47));let i=t.xor(r).mul(n);return i=i.xor(i.shru(47)),i=i.mul(n),i}function qr(e,t,n,r,i,a){i=i.add(e),a=Gr(a.add(i).add(r),21);let o=i;return i=i.add(t),i=i.add(n),a=a.add(Gr(i,44)),[i.add(r),a.add(o)]}function Jr(e,t,n,r){return qr(k(e,t),k(e,t+8),k(e,t+16),k(e,t+24),n,r)}function Yr(e,t=e.length){if(t>=8){let n=Vr.add(t*2),r=k(e,0).add(Vr),i=k(e,t-8);return Kr(Gr(i,37).mul(n).add(r),Gr(r,25).add(i).mul(n),n)}if(t>=4){let n=Vr.add(t*2);return Kr(Wr(e,0).shl(3).add(t),Wr(e,t-4),n)}if(t>0){let n=e[0],r=e[t>>1],i=e[t-1],a=n+(r<<8),o=t+(i<<2);return Hr(Vr.mul(a).xor(zr.mul(o))).mul(Vr)}return Vr}function Xr(e,t=e.length){let n=Vr.add(t*2),r=k(e,0).mul(Br),i=k(e,8),a=k(e,t-8).mul(n),o=k(e,t-16).mul(Vr);return Kr(Gr(r.add(i),43).add(Gr(a,30)).add(o),r.add(Gr(i.add(Vr),18)).add(a),n)}function Zr(e,t=e.length){let n=Vr.add(t*2),r=k(e,0).mul(Vr),i=k(e,8),a=k(e,t-8).mul(n),o=k(e,t-16).mul(Vr),s=Gr(r.add(i),43).add(Gr(a,30)).add(o),c=Kr(s,r.add(Gr(i.add(Vr),18)).add(a),n),l=k(e,16).mul(n),u=k(e,24),d=s.add(k(e,t-32)).mul(n),f=c.add(k(e,t-24)).mul(n);return Kr(Gr(l.add(u),43).add(Gr(d,30)).add(f),l.add(Gr(u.add(r),18)).add(d),n)}function Qr(e,t=e.length){let n=Lr.fromNumber(81,!0);if(t<=32)return t<=16?Yr(e,t):Xr(e,t);if(t<=64)return Zr(e,t);let r=n,i=n.mul(Br).add(113),a=Hr(i.mul(Vr).add(113)).mul(Vr),o=[Lr.UZERO,Lr.UZERO],s=[Lr.UZERO,Lr.UZERO];r=r.mul(Vr).add(k(e,0));let c=0,l=(t-1>>6)*64,u=l+(t-1&63)-63;do r=Gr(r.add(i).add(o[0]).add(k(e,c+8)),37).mul(Br),i=Gr(i.add(o[1]).add(k(e,c+48)),42).mul(Br),r=r.xor(s[1]),i=i.add(o[0]).add(k(e,c+40)),a=Gr(a.add(s[0]),33).mul(Br),o=Jr(e,c,o[1].mul(Br),r.add(s[0])),s=Jr(e,c+32,a.add(s[1]),i.add(k(e,c+16))),[a,r]=[r,a],c+=64;while(c!==l);let d=Br.add(a.and(255).shl(1));return c=u,s[0]=s[0].add(t-1&63),o[0]=o[0].add(s[0]),s[0]=s[0].add(o[0]),r=Gr(r.add(i).add(o[0]).add(k(e,c+8)),37).mul(d),i=Gr(i.add(o[1]).add(k(e,c+48)),42).mul(d),r=r.xor(s[1].mul(9)),i=i.add(o[0].mul(9).add(k(e,c+40))),a=Gr(a.add(s[0]),33).mul(d),o=Jr(e,c,o[1].mul(d),r.add(s[0])),s=Jr(e,c+32,a.add(s[1]),i.add(k(e,c+16))),[a,r]=[r,a],Kr(Kr(o[0],s[0],d).add(Hr(i).mul(zr)).add(a),Kr(o[1],s[1],d).add(r),d)}function $r(e,t){return t===`string`?ri(e):ti([e],t)}function ei(e,t){return e instanceof Float32Array&&t===`float32`||e instanceof Int32Array&&t===`int32`||e instanceof Uint8Array&&t===`bool`}function ti(e,t){if(t===`string`)throw Error(`Cannot convert a string[] to a TypedArray`);if(Array.isArray(e)&&(e=oi(e)),O().getBool(`DEBUG`)&&oe(e,t),ei(e,t))return e;if(t==null||t===`float32`||t===`complex64`)return new Float32Array(e);if(t===`int32`)return new Int32Array(e);if(t===`bool`){let t=new Uint8Array(e.length);for(let n=0;n<t.length;++n)Math.round(e[n])!==0&&(t[n]=1);return t}else throw Error(`Unknown data type ${t}`)}function ni(){return O().platform.now()}function ri(e,t=`utf-8`){return t||=`utf-8`,O().platform.encode(e,t)}function ii(e,t=`utf-8`){return t||=`utf-8`,O().platform.decode(e,t)}function ai(e){return O().platform.isTypedArray==null?Fr(e):O().platform.isTypedArray(e)}function oi(e,t=[],n=!1){if(t??=[],typeof e==`boolean`||typeof e==`number`||typeof e==`string`||Te(e)||e==null||ai(e)&&n)t.push(e);else if(Array.isArray(e)||ai(e))for(let r=0;r<e.length;++r)oi(e[r],t,n);else{let r=-1;for(let t of Object.keys(e))/^([1-9]+[0-9]*|0)$/.test(t)&&(r=Math.max(r,Number(t)));for(let i=0;i<=r;i++)oi(e[i],t,n)}return t}var si=class{constructor(e,t){this.backendTimer=e,this.logger=t,t??(this.logger=new li)}profileKernel(e,t,n){let r,i=()=>{r=n()},a,o=ni();if(this.backendTimer.timerAvailable())a=this.backendTimer.time(i);else{i();for(let e of r)e.dataSync();a=Promise.resolve({kernelMs:ni()-o})}if(O().getBool(`CHECK_COMPUTATION_FOR_ERRORS`))for(let t=0;t<r.length;t++){let n=r[t];n.data().then(t=>{ci(t,n.dtype,e)})}return{kernelName:e,outputs:r,inputs:t,timeMs:a.then(e=>e.kernelMs),extraInfo:a.then(e=>e.getExtraProfileInfo==null?``:e.getExtraProfileInfo())}}logKernelProfile(e){let{kernelName:t,outputs:n,timeMs:r,inputs:i,extraInfo:a}=e;n.forEach(e=>{Promise.all([e.data(),r,a]).then(n=>{this.logger.logKernelProfile(t,e,n[0],n[1],i,n[2])})})}};function ci(e,t,n){if(t!==`float32`)return!1;for(let t=0;t<e.length;t++){let r=e[t];if(isNaN(r)||!isFinite(r))return console.warn(`Found ${r} in the result of '${n}'`),!0}return!1}var li=class{logKernelProfile(e,t,n,r,i,a){let o=typeof r==`number`?E(`${r}ms`,9):r.error,s=E(e,25),c=t.rank,l=t.size,u=E(t.shape.toString(),14),d=``;for(let e in i){let n=i[e];if(n!=null){let r=n.shape||t.shape,i=r.length;d+=`${e}: ${i}D ${i>0?r:``} `}}console.log(`%c${s}\t%c${o}\t%c${c}D ${u}\t%c${l}\t%c${d}\t%c${a}`,`font-weight:bold`,`color:red`,`color:blue`,`color: orange`,`color: green`,`color: steelblue`)}};function ui(e,t,n){let r={},i={};for(let e=0;e<t.length;e++)r[t[e].id]=!0;for(let n=0;n<e.length;n++){let a=e[n],o=a.inputs;for(let e in o){let n=o[e],s=!1;for(let e=0;e<t.length;e++)if(r[n.id]){a.outputs.forEach(e=>r[e.id]=!0),s=!0,i[a.id]=!0;break}if(s)break}}let a={};a[n.id]=!0;let o={};for(let t=e.length-1;t>=0;t--){let n=e[t],r=n.inputs;for(let e=0;e<n.outputs.length;e++)if(a[n.outputs[e].id]){for(let e in r)a[r[e].id]=!0,o[n.id]=!0;break}}let s=[];for(let t=0;t<e.length;t++){let n=e[t];if(i[n.id]&&o[n.id]){let e={};for(let t in n.inputs){let i=n.inputs[t];r[i.id]&&(e[t]=i)}let t=Object.assign({},n);t.inputs=e,t.outputs=n.outputs,s.push(t)}}return s}function di(e,t,n,r){for(let i=t.length-1;i>=0;i--){let a=t[i],o=[];if(a.outputs.forEach(t=>{let n=e[t.id];n==null?o.push(null):o.push(n)}),a.gradient==null)throw Error(`Cannot compute gradient: gradient function not found for ${a.kernelName}.`);let s=a.gradient(o);for(let t in a.inputs){if(!(t in s))throw Error(`Cannot backprop through input ${t}. Available gradients found: ${Object.keys(s)}.`);let i=n(()=>s[t]());if(i.dtype!==`float32`)throw Error(`Error in gradient for op ${a.kernelName}. The gradient of input ${t} must have 'float32' dtype, but has '${i.dtype}'`);let o=a.inputs[t];if(!w(i.shape,o.shape))throw Error(`Error in gradient for op ${a.kernelName}. The gradient of input '${t}' has shape '${i.shape}', which does not match the shape of the input '${o.shape}'`);if(e[o.id]==null)e[o.id]=i;else{let t=e[o.id];e[o.id]=r(t,i),t.dispose()}}}}var fi=20,pi=3,mi=7;function hi(e,t,n,r){let i=_e(t),a=gi(e,t,n,i),o=t.length,s=yi(e,t,n,i,a),c=[`Tensor`];return r&&(c.push(`  dtype: ${n}`),c.push(`  rank: ${o}`),c.push(`  shape: [${t}]`),c.push(`  values:`)),c.push(s.map(e=>`    `+e).join(`
`)),c.join(`
`)}function gi(e,t,n,r){let i=S(t),a=r[r.length-1],o=Array(a).fill(0),s=t.length,c=n===`complex64`?bi(e):e;if(s>1)for(let e=0;e<i/a;e++){let t=e*a;for(let e=0;e<a;e++)o[e]=Math.max(o[e],_i(c[t+e],0,n).length)}return o}function _i(e,t,n){let r;return r=Array.isArray(e)?`${parseFloat(e[0].toFixed(mi))} + ${parseFloat(e[1].toFixed(mi))}j`:de(e)?`'${e}'`:n===`bool`?vi(e):parseFloat(e.toFixed(mi)).toString(),E(r,t)}function vi(e){return e===0?`false`:`true`}function yi(e,t,n,r,i,a=!0){let o=n===`complex64`?2:1,s=t[0],c=t.length;if(c===0)return n===`complex64`?[_i(bi(e)[0],0,n)]:n===`bool`?[vi(e[0])]:[e[0].toString()];if(c===1){if(s>fi){let t=pi*o,r=Array.from(e.slice(0,t)),a=Array.from(e.slice((s-pi)*o,s*o));return n===`complex64`&&(r=bi(r),a=bi(a)),[`[`+r.map((e,t)=>_i(e,i[t],n)).join(`, `)+`, ..., `+a.map((e,t)=>_i(e,i[s-pi+t],n)).join(`, `)+`]`]}return[`[`+(n===`complex64`?bi(e):Array.from(e)).map((e,t)=>_i(e,i[t],n)).join(`, `)+`]`]}let l=t.slice(1),u=r.slice(1),d=r[0]*o,f=[];if(s>fi){for(let t=0;t<pi;t++){let r=t*d,a=r+d;f.push(...yi(e.slice(r,a),l,n,u,i,!1))}f.push(`...`);for(let t=s-pi;t<s;t++){let r=t*d,a=r+d;f.push(...yi(e.slice(r,a),l,n,u,i,t===s-1))}}else for(let t=0;t<s;t++){let r=t*d,a=r+d;f.push(...yi(e.slice(r,a),l,n,u,i,t===s-1))}let p=c===2?`,`:``;f[0]=`[`+(s>0?f[0]+p:``);for(let e=1;e<f.length-1;e++)f[e]=` `+f[e]+p;let m=`,
`;for(let e=2;e<c;e++)m+=`
`;return f[f.length-1]=` `+f[f.length-1]+`]`+(a?``:m),f}function bi(e){let t=[];for(let n=0;n<e.length;n+=2)t.push([e[n],e[n+1]]);return t}var xi=class{constructor(e,t,n){if(this.dtype=t,this.shape=e.slice(),this.size=S(e),n!=null){let e=n.length;y(e===this.size,()=>`Length of values '${e}' does not match the size inferred by the shape '${this.size}'.`)}if(t===`complex64`)throw Error(`complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).`);this.values=n||ae(t,this.size),this.strides=_e(e)}set(e,...t){t.length===0&&(t=[0]),y(t.length===this.rank,()=>`The number of provided coordinates (${t.length}) must match the rank (${this.rank})`);let n=this.locToIndex(t);this.values[n]=e}get(...e){e.length===0&&(e=[0]);let t=0;for(let n of e){if(n<0||n>=this.shape[t]){let t=`Requested out of range element at ${e}.   Buffer shape=${this.shape}`;throw Error(t)}t++}let n=e[e.length-1];for(let t=0;t<e.length-1;++t)n+=this.strides[t]*e[t];return this.values[n]}locToIndex(e){if(this.rank===0)return 0;if(this.rank===1)return e[0];let t=e[e.length-1];for(let n=0;n<e.length-1;++n)t+=this.strides[n]*e[n];return t}indexToLoc(e){if(this.rank===0)return[];if(this.rank===1)return[e];let t=Array(this.shape.length);for(let n=0;n<t.length-1;++n)t[n]=Math.floor(e/this.strides[n]),e-=t[n]*this.strides[n];return t[t.length-1]=e,t}get rank(){return this.shape.length}toTensor(){return Si().makeTensor(this.values,this.shape,this.dtype)}},Si=null,Ci=null;function wi(e){Si=e}function Ti(e){Ci=e}var Ei=class{constructor(e,t,n,r){this.kept=!1,this.isDisposedInternal=!1,this.shape=e.slice(),this.dtype=t||`float32`,this.size=S(e),this.strides=_e(e),this.dataId=n,this.id=r,this.rankType=this.rank<5?this.rank.toString():`higher`}get rank(){return this.shape.length}async buffer(){let e=await this.data();return Ci.buffer(this.shape,this.dtype,e)}bufferSync(){return Ci.buffer(this.shape,this.dtype,this.dataSync())}async array(){let e=await this.data();return ye(this.shape,e,this.dtype===`complex64`)}arraySync(){return ye(this.shape,this.dataSync(),this.dtype===`complex64`)}async data(){this.throwIfDisposed();let e=Si().read(this.dataId);if(this.dtype===`string`){let t=await e;try{return t.map(e=>ii(e))}catch{throw Error(`Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().`)}}return e}dataToGPU(e){return this.throwIfDisposed(),Si().readToGPU(this.dataId,e)}dataSync(){this.throwIfDisposed();let e=Si().readSync(this.dataId);if(this.dtype===`string`)try{return e.map(e=>ii(e))}catch{throw Error(`Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().`)}return e}async bytes(){this.throwIfDisposed();let e=await Si().read(this.dataId);return this.dtype===`string`?e:new Uint8Array(e.buffer)}dispose(){this.isDisposed||(this.kerasMask&&this.kerasMask.dispose(),Si().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw Error(`Tensor is disposed.`)}print(e=!1){return Ci.print(this,e)}clone(){return this.throwIfDisposed(),Ci.clone(this)}toString(e=!1){return hi(this.dataSync(),this.shape,this.dtype,e)}cast(e){return this.throwIfDisposed(),Ci.cast(this,e)}variable(e=!0,t,n){return this.throwIfDisposed(),Si().makeVariable(this,e,t,n)}};Object.defineProperty(Ei,Symbol.hasInstance,{value:e=>!!e&&e.data!=null&&e.dataSync!=null&&e.throwIfDisposed!=null});function Di(){return Ie(`Tensor`,()=>Ei)}Di();var Oi=class extends Ei{constructor(e,t,n,r){super(e.shape,e.dtype,e.dataId,r),this.trainable=t,this.name=n}assign(e){if(e.dtype!==this.dtype)throw Error(`dtype of the new value (${e.dtype}) and previous value (${this.dtype}) must match`);if(!w(e.shape,this.shape))throw Error(`shape of the new value (${e.shape}) and previous value (${this.shape}) must match`);Si().disposeTensor(this),this.dataId=e.dataId,Si().incRef(this,null)}dispose(){Si().disposeVariable(this),this.isDisposedInternal=!0}};Object.defineProperty(Oi,Symbol.hasInstance,{value:e=>e instanceof Ei&&e.assign!=null&&e.assign instanceof Function});var ki;(function(e){e.R0=`R0`,e.R1=`R1`,e.R2=`R2`,e.R3=`R3`,e.R4=`R4`,e.R5=`R5`,e.R6=`R6`})(ki||={});var Ai;(function(e){e.float32=`float32`,e.int32=`int32`,e.bool=`int32`,e.complex64=`complex64`})(Ai||={});var ji;(function(e){e.float32=`float32`,e.int32=`int32`,e.bool=`bool`,e.complex64=`complex64`})(ji||={});var Mi;(function(e){e.float32=`float32`,e.int32=`float32`,e.bool=`float32`,e.complex64=`complex64`})(Mi||={});var Ni;(function(e){e.float32=`complex64`,e.int32=`complex64`,e.bool=`complex64`,e.complex64=`complex64`})(Ni||={});var Pi={float32:Mi,int32:Ai,bool:ji,complex64:Ni};function Fi(e,t){if(e===`string`||t===`string`){if(e===`string`&&t===`string`)return`string`;throw Error(`Can not upcast ${e} with ${t}`)}return Pi[e][t]}function Ii(e){return Fi(e,`int32`)}function Li(e){return typeof e==`object`&&!!e&&`texture`in e&&e.texture instanceof WebGLTexture}function Ri(e){return typeof GPUBuffer<`u`&&typeof e==`object`&&!!e&&`buffer`in e&&e.buffer instanceof GPUBuffer}function A(e,t){if(e.dtype===t.dtype)return[e,t];let n=Fi(e.dtype,t.dtype);return[e.cast(n),t.cast(n)]}function zi(e,t){y(e.dtype===t.dtype,()=>`The dtypes of the first(${e.dtype}) and second(${t.dtype}) input must match`)}function Bi(e){let t=[];return Vi(e,t,new Set),t}function Vi(e,t,n){if(e==null)return;if(e instanceof Ei){t.push(e);return}if(!Hi(e))return;let r=e;for(let e in r){let i=r[e];n.has(i)||(n.add(i),Vi(i,t,n))}}function Hi(e){return Array.isArray(e)||typeof e==`object`}function Ui(e){return e.kernelName!=null}var Wi=class{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(e=>e.name)))}}}dispose(){for(let e in this.registeredVariables)this.registeredVariables[e].dispose()}},Gi=class e{constructor(e){this.ENV=e,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new Wi}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;let e=this.getSortedBackends();for(let t=0;t<e.length;t++){let n=e[t];if(await this.initializeBackend(n).success){await this.setBackend(n);return}}throw Error(`Could not initialize any backends, all backend initializations failed.`)}get backend(){if(this.pendingBackendInit!=null)throw Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){let{name:e,asyncInit:t}=this.initializeBackendsAndReturnBest();if(t)throw Error(`The highest priority backend '${e}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(e)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(e){if(!(e in this.registry))if(e in this.registryFactory){let{asyncInit:t}=this.initializeBackend(e);if(t)return null}else return null;return this.registry[e]}findBackendFactory(e){return e in this.registryFactory?this.registryFactory[e].factory:null}registerBackend(e,t,n=1){return e in this.registryFactory?(Er(`${e} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[e]={factory:t,priority:n},!0)}async setBackend(e){if(this.registryFactory[e]==null)throw Error(`Backend name '${e}' not found in registry`);if(this.backendName=e,this.registry[e]==null){this.backendInstance=null;let{success:t,asyncInit:n}=this.initializeBackend(e);if(!(n?await t:t))return!1}return this.backendInstance=this.registry[e],this.setupRegisteredKernels(),this.profiler=new si(this.backendInstance),!0}setupRegisteredKernels(){Mr(this.backendName).forEach(e=>{e.setupFunc!=null&&e.setupFunc(this.backendInstance)})}disposeRegisteredKernels(e){Mr(e).forEach(t=>{t.disposeFunc!=null&&t.disposeFunc(this.registry[e])})}initializeBackend(e){let t=this.registryFactory[e];if(t==null)throw Error(`Cannot initialize backend ${e}, no registration found.`);try{let n=t.factory();if(n&&!(n instanceof p)&&typeof n.then==`function`){let t=++this.pendingBackendInitId,r=n.then(n=>t<this.pendingBackendInitId?!1:(this.registry[e]=n,this.pendingBackendInit=null,!0)).catch(n=>t<this.pendingBackendInitId?!1:(this.pendingBackendInit=null,Er(`Initialization of backend ${e} failed`),Er(n.stack||n.message),!1));return this.pendingBackendInit=r,{success:r,asyncInit:!0}}else return this.registry[e]=n,{success:!0,asyncInit:!1}}catch(t){return Er(`Initialization of backend ${e} failed`),Er(t.stack||t.message),{success:!1,asyncInit:!1}}}removeBackend(e){if(!(e in this.registryFactory))throw Error(`${e} backend not found in registry`);this.backendName===e&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,e in this.registry&&(this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e]),delete this.registryFactory[e],this.backendName===e&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw Error(`No backend found in registry.`);return Object.keys(this.registryFactory).sort((e,t)=>this.registryFactory[t].priority-this.registryFactory[e].priority)}initializeBackendsAndReturnBest(){let e=this.getSortedBackends();for(let t=0;t<e.length;t++){let n=e[t],{success:r,asyncInit:i}=this.initializeBackend(n);if(i||r)return{name:n,asyncInit:i}}throw Error(`Could not initialize any backends, all backend initializations failed.`)}moveData(e,t){let n=this.state.tensorInfo.get(t),r=n.backend,i=this.readSync(t),a=r.refCount(t);r.disposeData(t,!0),n.backend=e,e.move(t,i,n.shape,n.dtype,a),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(e,t){let n=null;if(t==null){if(typeof e!=`function`)throw Error(`Please provide a function to tidy()`);t=e}else{if(typeof e!=`string`&&!(e instanceof String))throw Error(`When calling with two arguments, the first argument to tidy() must be a string`);if(typeof t!=`function`)throw Error(`When calling with two arguments, the 2nd argument to tidy() must be a function`);n=e}let r;return this.scopedRun(()=>this.startScope(n),()=>this.endScope(r),()=>(r=t(),r instanceof Promise&&console.error(`Cannot return a Promise inside of tidy.`),r))}scopedRun(e,t,n){e();try{let e=n();return t(),e}catch(e){throw t(),e}}nextTensorId(){return e.nextTensorId++}nextVariableId(){return e.nextVariableId++}clone(e){let t=j.runKernel(Bt,{x:e}),n={x:e};return this.addTapeNode(this.state.activeScope.name,n,[t],e=>({x:()=>{let t={x:e};return j.runKernel(nt,t,{dtype:`float32`})}}),[],{}),t}runKernel(e,t,n){if(this.backendName??this.backend,Ar(e,this.backendName)==null)throw Error(`Kernel '${e}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:e,inputs:t,attrs:n})}shouldCheckForMemLeaks(){return this.ENV.getBool(`IS_TEST`)}checkKernelForMemLeak(e,t,n){let r=this.backend.numDataIds(),i=0;n.forEach(e=>{i+=e.dtype===`complex64`?3:1});let a=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],o=r-t-i-a;if(o>0)throw Error(`Backend '${this.backendName}' has an internal memory leak (${o} data ids) after running '${e}'`)}runKernelFunc(e){let t,n=[],r=this.isTapeOn(),i=this.state.numBytes,a=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let o;this.backendName??this.backend;let s,c=Ui(e)?e.kernelName:this.state.activeScope==null?``:this.state.activeScope.name;if(Ui(e)){let{kernelName:t,inputs:i,attrs:a}=e;this.backendName??this.backend;let c=Ar(t,this.backendName);y(c!=null,()=>`Cannot find registered kernel '${t}' for backend '${this.backendName}'`),o=()=>{let e=this.backend.numDataIds();s=c.kernelFunc({inputs:i,attrs:a,backend:this.backend});let o=Array.isArray(s)?s:[s];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(t,e,o);let l=o.map(e=>e.rank==null?this.makeTensorFromTensorInfo(e):e);if(r){let e=this.getTensorsForGradient(t,i,l);n=this.saveTensorsForBackwardMode(e)}return l}}else{let{forwardFunc:t}=e,i=e=>{r&&(n=e.map(e=>this.keep(this.clone(e))))};o=()=>{let e=this.backend.numDataIds();s=this.tidy(()=>t(this.backend,i));let n=Array.isArray(s)?s:[s];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(c,e,n),n}}let{inputs:l,attrs:u}=e,d=Ui(e)?null:e.backwardsFunc,f;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool(`DEBUG`)&&!this.state.profiling?t=o():(f=this.profiler.profileKernel(c,l,()=>o()),this.ENV.getBool(`DEBUG`)&&this.profiler.logKernelProfile(f),t=f.outputs)}),r&&this.addTapeNode(c,l,t,d,n,u),this.state.profiling&&this.state.activeProfile.kernels.push({name:c,bytesAdded:this.state.numBytes-i,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-a,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(l).map(e=>l[e]==null?null:l[e].shape),outputShapes:t.map(e=>e.shape),kernelTimeMs:f.timeMs,extraInfo:f.extraInfo}),Array.isArray(s)?t:t[0]}saveTensorsForBackwardMode(e){return e.map(e=>this.keep(this.clone(e)))}getTensorsForGradient(e,t,n){let r=jr(e);if(r!=null){let e=r.inputsToSave||[],i=r.outputsToSave||[],a;r.saveAllInputs?(y(Array.isArray(t),()=>`saveAllInputs is true, expected inputs to be an array.`),a=Object.keys(t).map(e=>t[e])):a=e.map(e=>t[e]);let o=n.filter((e,t)=>i[t]);return a.concat(o)}return[]}makeTensor(e,t,n,r){if(e==null)throw Error(`Values passed to engine.makeTensor() are null`);n||=`float32`,r||=this.backend;let i=e;n===`string`&&de(e[0])&&(i=e.map(e=>ri(e)));let a=r.write(i,t,n),o=new Ei(t,n,a,this.nextTensorId());if(this.trackTensor(o,r),n===`string`){let e=this.state.tensorInfo.get(a),t=ue(i);this.state.numBytes+=t-e.bytes,e.bytes=t}return o}makeTensorFromDataId(e,t,n,r){n||=`float32`;let i={dataId:e,shape:t,dtype:n};return this.makeTensorFromTensorInfo(i,r)}makeTensorFromTensorInfo(e,t){let{dataId:n,shape:r,dtype:i}=e,a=new Ei(r,i,n,this.nextTensorId());return this.trackTensor(a,t),a}makeVariable(e,t=!0,n,r){n||=this.nextVariableId().toString(),r!=null&&r!==e.dtype&&(e=e.cast(r));let i=new Oi(e,t,n,this.nextTensorId());if(this.state.registeredVariables[i.name]!=null)throw Error(`Variable with name ${i.name} was already registered`);return this.state.registeredVariables[i.name]=i,this.incRef(i,this.backend),i}trackTensor(e,t){this.state.numTensors++,e.dtype===`string`&&this.state.numStringTensors++;let n=0;e.dtype!==`complex64`&&e.dtype!==`string`&&(n=e.size*le(e.dtype)),this.state.numBytes+=n,this.state.tensorInfo.has(e.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(e.dataId,{backend:t||this.backend,dtype:e.dtype,shape:e.shape,bytes:n})),e instanceof Oi||this.track(e)}incRef(e,t){this.trackTensor(e,t),this.backend.incRef(e.dataId)}removeDataId(e,t){this.state.tensorInfo.has(e)&&this.state.tensorInfo.get(e).backend===t&&(this.state.tensorInfo.delete(e),this.state.numDataBuffers--)}disposeTensor(e){if(!this.state.tensorInfo.has(e.dataId))return;let t=this.state.tensorInfo.get(e.dataId);if(this.state.numTensors--,e.dtype===`string`&&(this.state.numStringTensors--,this.state.numBytes-=t.bytes),e.dtype!==`complex64`&&e.dtype!==`string`){let t=e.size*le(e.dtype);this.state.numBytes-=t}t.backend.disposeData(e.dataId)&&this.removeDataId(e.dataId,t.backend)}disposeVariables(){for(let e in this.state.registeredVariables){let t=this.state.registeredVariables[e];this.disposeVariable(t)}}disposeVariable(e){this.disposeTensor(e),this.state.registeredVariables[e.name]!=null&&delete this.state.registeredVariables[e.name]}memory(){let e=this.backend.memory();return e.numTensors=this.state.numTensors,e.numDataBuffers=this.state.numDataBuffers,e.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(e.unreliable=!0,e.reasons??=[],e.reasons.push(`Memory usage by string tensors is approximate (2 bytes per character)`)),e}async profile(e){this.state.profiling=!0;let t=this.state.numBytes,n=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await e(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(e=>e.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-t,this.state.activeProfile.newTensors=this.state.numTensors-n;for(let e of this.state.activeProfile.kernels)e.kernelTimeMs=await e.kernelTimeMs,e.extraInfo=await e.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(e,t,n,r,i,a){let o={id:this.state.nextTapeNodeId++,kernelName:e,inputs:t,outputs:n,saved:i},s=jr(e);s!=null&&(r=s.gradFunc),r!=null&&(o.gradient=e=>(e=e.map((e,t)=>{if(e==null){let e=n[t],r=xe(e.size,e.dtype);return this.makeTensor(r,e.shape,e.dtype)}return e}),r(e.length>1?e:e[0],i,a))),this.state.activeTape.push(o)}keep(e){return e.kept=!0,e}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(e){let t={track:[],name:`unnamed scope`,id:this.state.nextScopeId++};e&&(t.name=e),this.state.scopeStack.push(t),this.state.activeScope=t}endScope(e){let t=Bi(e),n=new Set(t.map(e=>e.id));for(let e=0;e<this.state.activeScope.track.length;e++){let t=this.state.activeScope.track[e];!t.kept&&!n.has(t.id)&&t.dispose()}let r=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],t.forEach(e=>{!e.kept&&e.scopeId===r.id&&this.track(e)})}gradients(e,t,n,r=!1){if(y(t.length>0,()=>`gradients() received an empty list of xs.`),n!=null&&n.dtype!==`float32`)throw Error(`dy must have 'float32' dtype, but has '${n.dtype}'`);let i=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy(`forward`,e));y(i instanceof Ei,()=>`The result y returned by f() must be a tensor.`);let a=ui(this.state.activeTape,t,i);if(!r&&a.length===0&&t.length>0)throw Error(`Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.`);return this.tidy(`backward`,()=>{let e={};e[i.id]=n??Ki(i.shape),di(e,a,e=>this.tidy(e),Ji);let r=t.map(t=>e[t.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(e=>{for(let t of e.saved)t.dispose()}),this.state.activeTape=null),{value:i,grads:r}})}customGrad(e){return y(he(e),()=>`The f passed in customGrad(f) must be a function.`),(...t)=>{y(t.every(e=>e instanceof Ei),()=>`The args passed in customGrad(f)(x1, x2,...) must all be tensors`);let n,r={};return t.forEach((e,t)=>{r[t]=e}),this.runKernelFunc({forwardFunc:(r,i)=>(n=e(...t,i),y(n.value instanceof Ei,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),y(he(n.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),n.value),backwardsFunc:(e,r)=>{let i=n.gradFunc(e,r),a=Array.isArray(i)?i:[i];y(a.length===t.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),y(a.every(e=>e instanceof Ei),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");let o={};return a.forEach((e,t)=>{o[t]=()=>e}),o},inputs:r})}}readSync(e){return this.state.tensorInfo.get(e).backend.readSync(e)}read(e){return this.state.tensorInfo.get(e).backend.read(e)}readToGPU(e,t){return this.state.tensorInfo.get(e).backend.readToGPU(e,t)}async time(e){let t=ni(),n=await this.backend.time(e);return n.wallMs=ni()-t,n}track(e){return this.state.activeScope!=null&&(e.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(e)),e}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new Wi;for(let e in this.registry)this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}};Gi.nextTensorId=0,Gi.nextVariableId=0;function Ki(e){let t=be(S(e),`float32`);return j.makeTensor(t,e,`float32`)}function qi(){let e=Pe();return e._tfengine??=new Gi(new De(e)),Me(e._tfengine.ENV),wi(()=>e._tfengine),e._tfengine}var j=qi();function Ji(e,t){let n={a:e,b:t};return j.runKernel(`Add`,n)}function Yi(){return typeof navigator<`u`&&navigator!=null}var Xi;function Zi(e){if(Xi!==void 0)return Xi;if(e||Yi()){if(e||=navigator,e.product===`ReactNative`)return!0;let t=e.userAgent||e.vendor||(typeof window<`u`?window.opera:``);if(!t){let t=e;return t.userAgentData&&t.userAgentData.mobile}return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(t)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(t.substr(0,4))}return!1}function Qi(){return typeof window<`u`&&window.document!=null||typeof WorkerGlobalScope<`u`}var $i=O();$i.registerFlag(`DEBUG`,()=>!1,e=>{e&&console.warn(`Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.`)}),$i.registerFlag(`IS_BROWSER`,()=>Qi()),$i.registerFlag(`IS_NODE`,()=>typeof process<`u`&&process.versions!==void 0&&process.versions.node!==void 0),$i.registerFlag(`IS_CHROME`,()=>typeof navigator<`u`&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor)),$i.registerFlag(`IS_SAFARI`,()=>typeof navigator<`u`&&navigator!=null&&navigator.userAgent!=null&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor)),$i.registerFlag(`PROD`,()=>!1),$i.registerFlag(`TENSORLIKE_CHECK_SHAPE_CONSISTENCY`,()=>$i.getBool(`DEBUG`)),$i.registerFlag(`DEPRECATION_WARNINGS_ENABLED`,()=>!0),$i.registerFlag(`IS_TEST`,()=>!1),$i.registerFlag(`CHECK_COMPUTATION_FOR_ERRORS`,()=>$i.getBool(`DEBUG`)),$i.registerFlag(`WRAP_TO_IMAGEBITMAP`,()=>!1),$i.registerFlag(`CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU`,()=>!1),$i.registerFlag(`USE_SETTIMEOUTCUSTOM`,()=>!1);function ea(e,t){let n=e;if(ai(e))return t===`string`?[]:[e.length];if(Li(e)){let t=e.channels||`RGBA`;return[e.height,e.width*t.length]}else if(Ri(e))return[e.buffer.size/(t==null?4:le(t))];if(!Array.isArray(e))return[];let r=[];for(;Array.isArray(n)||ai(n)&&t!==`string`;)r.push(n.length),n=n[0];return Array.isArray(e)&&O().getBool(`TENSORLIKE_CHECK_SHAPE_CONSISTENCY`)&&ta(e,r,[]),r}function ta(e,t,n){if(n||=[],!Array.isArray(e)&&!ai(e)){y(t.length===0,()=>`Element arr[${n.join(`][`)}] is a primitive, but should be an array/TypedArray of ${t[0]} elements`);return}y(t.length>0,()=>`Element arr[${n.join(`][`)}] should be a primitive, but is an array of ${e.length} elements`),y(e.length===t[0],()=>`Element arr[${n.join(`][`)}] should have ${t[0]} elements, but has ${e.length} elements`);let r=t.slice(1);for(let t=0;t<e.length;++t)ta(e[t],r,n.concat(t))}function na(e,t,n,r){if(e!==`string_or_numeric`){if(e==null)throw Error(`Expected dtype cannot be null.`);if(e!==`numeric`&&e!==t||e===`numeric`&&t===`string`)throw Error(`Argument '${n}' passed to '${r}' must be ${e} tensor, but got ${t} tensor`)}}function M(e,t,n,r=`numeric`){if(e instanceof Di())return na(r,e.dtype,t,n),e;let i=me(e);if(i!==`string`&&[`bool`,`int32`,`float32`].indexOf(r)>=0&&(i=r),na(r,i,t,n),e==null||!ai(e)&&!Array.isArray(e)&&typeof e!=`number`&&typeof e!=`boolean`&&typeof e!=`string`){let r=e==null?`null`:e.constructor.name;throw Error(`Argument '${t}' passed to '${n}' must be a Tensor or TensorLike, but got '${r}'`)}let a=ea(e,i);!ai(e)&&!Array.isArray(e)&&(e=[e]);let o=i===`string`?oi(e,[],!0):ti(e,i);return j.makeTensor(o,a,i)}function ra(e,t,n,r=`numeric`){if(!Array.isArray(e))throw Error(`Argument ${t} passed to ${n} must be a \`Tensor[]\` or \`TensorLike[]\``);return e.map((e,i)=>M(e,`${t}[${i}]`,n,r))}var ia=`__op`;function N(e){let t=Object.keys(e);if(t.length!==1)throw Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${t.length} keys.`);let n=t[0],r=e[n];n.endsWith(`_`)&&(n=n.substring(0,n.length-1)),n+=ia;let i=(...e)=>{j.startScope(n);try{let t=r(...e);return Te(t)&&console.error(`Cannot return a Promise inside of tidy.`),j.endScope(t),t}catch(e){throw j.endScope(null),e}};return Object.defineProperty(i,`name`,{value:n,configurable:!0}),i}function aa(e,t){let n=M(e,`real`,`complex`),r=M(t,`imag`,`complex`);b(n.shape,r.shape,`real and imag shapes, ${n.shape} and ${r.shape}, must match in call to tf.complex().`);let i={real:n,imag:r};return j.runKernel(at,i)}var oa=N({complex_:aa});function sa(e,t,n,r){if(r==null)r=me(e);else if(r===`complex64`)throw Error(`Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).`);if(Ri(e)||Li(e)){if(r!==`float32`&&r!==`int32`)throw Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${r}.`);return j.backend.createTensorFromGPUData(e,t||n,r)}if(!ai(e)&&!Array.isArray(e)&&typeof e!=`number`&&typeof e!=`boolean`&&typeof e!=`string`)throw Error(`values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray`);if(t!=null){Se(t);let e=S(t),r=S(n);y(e===r,()=>`Based on the provided shape, [${t}], the tensor should have ${e} values but has ${r}`);for(let e=0;e<n.length;++e){let r=n[e],i=e===n.length-1?r!==S(t.slice(e)):!0;y(n[e]===t[e]||!i,()=>`Error creating a new Tensor. Inferred shape (${n}) does not match the provided shape (${t}). `)}}return!ai(e)&&!Array.isArray(e)&&(e=[e]),t||=n,e=r===`string`?oi(e,[],!0):ti(e,r),j.makeTensor(e,t,r)}function ca(e,t,n){return sa(e,t,ea(e,n),n)}var la={float32:4,float16:2,int32:4,uint16:2,uint8:1,bool:1,complex64:8},ua=class e{static join(t){return new e(t).slice()}constructor(e){if(this.shards=[],this.previousShardIndex=0,e==null||(e instanceof Array||(e=[e]),e=e.map(e=>ai(e)?e.buffer:e),e.length===0))return;this.bufferUniformSize=e[0].byteLength;let t=0;for(let n=0;n<e.length;n++){let r=e[n];n!==e.length-1&&r.byteLength!==this.bufferUniformSize&&(this.bufferUniformSize=void 0);let i=t+r.byteLength;this.shards.push({buffer:r,start:t,end:i}),t=i}this.shards.length===0&&(this.byteLength=0),this.byteLength=this.shards[this.shards.length-1].end}slice(e=0,t=this.byteLength){if(this.shards.length===0||(e=isNaN(Number(e))?0:e,t=isNaN(Number(t))?0:t,e=Math.max(0,e),t=Math.min(this.byteLength,t),t<=e))return new ArrayBuffer(0);let n=this.findShardForByte(e);if(n===-1)throw Error(`Could not find start shard for byte ${e}`);let r=t-e,i=new ArrayBuffer(r),a=new Uint8Array(i),o=0;for(let r=n;r<this.shards.length;r++){let n=this.shards[r],i=e+o-n.start,s=o,c=Math.min(t,n.end)-n.start,l=new Uint8Array(n.buffer,i,c-i);if(a.set(l,s),o+=l.length,t<n.end)break}return i}findShardForByte(e){if(this.shards.length===0||e<0||e>=this.byteLength)return-1;if(this.bufferUniformSize!=null)return this.previousShardIndex=Math.floor(e/this.bufferUniformSize),this.previousShardIndex;function t(t){return e<t.start?-1:+(e>=t.end)}if(t(this.shards[this.previousShardIndex])===0)return this.previousShardIndex;let n=da(this.shards,t);return n===-1?-1:(this.previousShardIndex=n,this.previousShardIndex)}};function da(e,t){let n=0,r=e.length;for(;n<=r;){let i=Math.floor((r-n)/2)+n,a=t(e[i]);if(a===0)return i;a<0?r=i:n=i+1}return-1}function fa(){return j}function P(e,t){return j.tidy(e,t)}function F(e){Bi(e).forEach(e=>e.dispose())}function pa(e){return j.keep(e)}function ma(){return j.backendName}function ha(e,t,n=1){return j.registerBackend(e,t,n)}function ga(){return j.backend}var _a=4;async function va(e,t){let n=[],r=[],i=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);for(let a=0;a<i.length;++a){let o=i[a],s=Array.isArray(e)?e[a].tensor:e[o];if(s.dtype!==`float32`&&s.dtype!==`int32`&&s.dtype!==`bool`&&s.dtype!==`string`&&s.dtype!==`complex64`)throw Error(`Unsupported dtype in weight '${o}': ${s.dtype}`);let c={name:o,shape:s.shape,dtype:s.dtype};if(s.dtype===`string`){let e=new Promise(async e=>{let t=await s.bytes(),n=t.reduce((e,t)=>e+t.length,0)+_a*t.length,r=new Uint8Array(n),i=0;for(let e=0;e<t.length;e++){let n=t[e],a=new Uint8Array(new Uint32Array([n.length]).buffer);r.set(a,i),i+=_a,r.set(n,i),i+=n.length}e(r)});r.push(e)}else r.push(s.data());t!=null&&(c.group=t),n.push(c)}return{data:Ta(await Promise.all(r)),specs:n}}function ya(e,t){let n=new ua(e),r={},i=0;for(let e of t){let t=ba(e,(e,t)=>n.slice(i+e,i+t));r[e.name]=Sa(e,n.slice(i,i+t)),i+=t}return r}function ba(e,t){let n=S(e.shape),r;if(`quantization`in e)r=la[e.quantization.dtype];else if(e.dtype===`string`){let e=0;for(let r=0;r<n;r++)e+=_a+new Uint32Array(t(e,e+_a))[0];return e}else r=la[e.dtype];return n*r}async function xa(e,t){let n=S(e.shape),r;if(`quantization`in e)r=la[e.quantization.dtype];else if(e.dtype===`string`){let e=0;for(let r=0;r<n;r++)e+=_a+new Uint32Array(await t(e,e+_a))[0];return e}else r=la[e.dtype];return n*r}function Sa(e,t){let n=e.name,r=e.dtype,i=e.shape,a=S(i),o,s=0;if(`quantization`in e){let i=e.quantization;if(i.dtype===`uint8`||i.dtype===`uint16`){if(!(`min`in i&&`scale`in i))throw Error(`Weight ${e.name} with quantization ${i.dtype} doesn't have corresponding metadata min and scale.`)}else if(i.dtype===`float16`){if(r!==`float32`)throw Error(`Weight ${e.name} is quantized with ${i.dtype} which only supports weights of type float32 not ${r}.`)}else throw Error(`Weight ${e.name} has unknown quantization dtype ${i.dtype}. Supported quantization dtypes are: 'uint8', 'uint16', and 'float16'.`);let c=la[i.dtype],l=i.dtype===`uint8`?new Uint8Array(t):new Uint16Array(t);if(r===`float32`)if(i.dtype===`uint8`||i.dtype===`uint16`){o=new Float32Array(l.length);for(let e=0;e<l.length;e++){let t=l[e];o[e]=t*i.scale+i.min}}else if(i.dtype===`float16`)o=Ba()(l);else throw Error(`Unsupported quantization type ${i.dtype} for weight type float32.`);else if(r===`int32`){if(i.dtype!==`uint8`&&i.dtype!==`uint16`)throw Error(`Unsupported quantization type ${i.dtype} for weight type int32.`);o=new Int32Array(l.length);for(let e=0;e<l.length;e++){let t=l[e];o[e]=Math.round(t*i.scale+i.min)}}else throw Error(`Unsupported dtype in weight '${n}': ${r}`);s+=a*c}else if(r===`string`){let n=S(e.shape);o=[];for(let e=0;e<n;e++){let e=new Uint32Array(t.slice(s,s+_a))[0];s+=_a;let n=new Uint8Array(t.slice(s,s+e));o.push(n),s+=e}}else{let e=la[r];if(r===`float32`)o=new Float32Array(t);else if(r===`int32`)o=new Int32Array(t);else if(r===`bool`)o=new Uint8Array(t);else if(r===`complex64`){o=new Float32Array(t);let e=new Float32Array(o.length/2),n=new Float32Array(o.length/2);for(let t=0;t<e.length;t++)e[t]=o[t*2],n[t]=o[t*2+1];let r=ca(e,i,`float32`),a=ca(n,i,`float32`),s=oa(r,a);return r.dispose(),a.dispose(),s}else throw Error(`Unsupported dtype in weight '${n}': ${r}`);s+=a*e}return ca(o,i,r)}async function Ca(e,t,n){let r=new Uint8Array(t);for(;r.byteLength<n;){let{done:t,value:i}=await e.read();if(t&&i==null){let e=n-r.byteLength;throw Error(`Reader is done but ${e} bytes are still expected`)}let a=new Uint8Array(r.length+i.byteLength);a.set(r,0),a.set(new Uint8Array(i),r.length),r=a}return r.buffer}async function wa(e,t){let n={},r=e.getReader(),i=new ArrayBuffer(0);for(let e of t){let t=await xa(e,async(e,t)=>(i=await Ca(r,i,t),i.slice(e,t)));i=await Ca(r,i,t);let a=i.slice(0,t);i=i.slice(t);let o=Sa(e,a);if(n[e.name]=o,ma()===`webgpu`){let e=ga();`uploadToGPU`in e&&S(o.shape)>=O().get(`WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD`)&&e.uploadToGPU(o.dataId)}}return n}function Ta(e){if(e===null)throw Error(`Invalid input value: ${JSON.stringify(e)}`);let t=0,n=[];e.forEach(e=>{if(t+=e.byteLength,n.push(e.byteLength===e.buffer.byteLength?e:new e.constructor(e)),!(e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array))throw Error(`Unsupported TypedArray subtype: ${e.constructor.name}`)});let r=new Uint8Array(t),i=0;return n.forEach(e=>{r.set(new Uint8Array(e.buffer),i),i+=e.byteLength}),r.buffer}var Ea=typeof Buffer<`u`&&(typeof Blob>`u`||typeof atob>`u`||typeof btoa>`u`);function Da(e){return Ea?Buffer.byteLength(e,`utf8`):new Blob([e]).size}function Oa(e){if(Ea)return Buffer.from(e).toString(`base64`);let t=new Uint8Array(e),n=``;for(let e=0,r=t.length;e<r;e++)n+=String.fromCharCode(t[e]);return btoa(n)}function ka(e){if(Ea){let t=Buffer.from(e,`base64`);return t.buffer.slice(t.byteOffset,t.byteOffset+t.byteLength)}let t=atob(e),n=new Uint8Array(t.length);for(let e=0;e<t.length;++e)n.set([t.charCodeAt(e)],e);return n.buffer}function Aa(e){return ua.join(e)}function ja(e){for(e=e.trim();e.endsWith(`/`);)e=e.slice(0,e.length-1);let t=e.split(`/`);return t[t.length-1]}function Ma(e,t){let n={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,weightsManifest:t};return e.signature!=null&&(n.signature=e.signature),e.userDefinedMetadata!=null&&(n.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(n.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(n.initializerSignature=e.initializerSignature),e.trainingConfig!=null&&(n.trainingConfig=e.trainingConfig),n}function Na(e,t,n){let r={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy};if(e.trainingConfig!=null&&(r.trainingConfig=e.trainingConfig),e.weightsManifest!=null){if(!t)throw Error(`modelJSON has weightsManifest but weightSpecs is null`);if(!n)throw Error(`modelJSON has weightsManifest but weightData is null`);r.weightSpecs=t,r.weightData=n}return e.signature!=null&&(r.signature=e.signature),e.userDefinedMetadata!=null&&(r.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(r.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(r.initializerSignature=e.initializerSignature),r}async function Pa(e,t){let n,r;return e.weightsManifest!=null&&([n,r]=await t(e.weightsManifest)),Na(e,n,r)}function Fa(e){if(e.modelTopology instanceof ArrayBuffer)throw Error(`Expected JSON model topology, received ArrayBuffer.`);return{dateSaved:new Date,modelTopologyType:`JSON`,modelTopologyBytes:e.modelTopology==null?0:Da(JSON.stringify(e.modelTopology)),weightSpecsBytes:e.weightSpecs==null?0:Da(JSON.stringify(e.weightSpecs)),weightDataBytes:e.weightData==null?0:new ua(e.weightData).byteLength}}function Ia(e){let t=[];for(let n of e)t.push(...n.weights);return t}function La(){let e=e=>{let t=e<<13,n=0;for(;!(t&8388608);)n-=8388608,t<<=1;return t&=-8388609,n+=947912704,t|n},t=new Uint32Array(2048);t[0]=0;for(let n=1;n<1024;n++)t[n]=e(n);for(let e=1024;e<2048;e++)t[e]=939524096+(e-1024<<13);return t}function Ra(){let e=new Uint32Array(64);e[0]=0,e[31]=1199570944,e[32]=2147483648,e[63]=3347054592;for(let t=1;t<31;t++)e[t]=t<<23;for(let t=33;t<63;t++)e[t]=2147483648+(t-32<<23);return e}function za(){let e=new Uint32Array(64);for(let t=0;t<64;t++)e[t]=1024;return e[0]=e[32]=0,e}function Ba(){let e=La(),t=Ra(),n=za();return r=>{let i=new ArrayBuffer(4*r.length),a=new Uint32Array(i);for(let i=0;i<r.length;i++){let o=r[i];a[i]=e[n[o>>10]+(o&1023)]+t[o>>10]}return new Float32Array(i)}}var Va=class e{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return e.instance??=new e,e.instance}static registerSaveRouter(t){e.getInstance().saveRouters.push(t)}static registerLoadRouter(t){e.getInstance().loadRouters.push(t)}static getSaveHandlers(t){return e.getHandlers(t,`save`)}static getLoadHandlers(t,n){return e.getHandlers(t,`load`,n)}static getHandlers(t,n,r){let i=[];return(n===`load`?e.getInstance().loadRouters:e.getInstance().saveRouters).forEach(e=>{let n=e(t,r);n!==null&&i.push(n)}),i}},Ha=e=>Va.registerSaveRouter(e),Ua=e=>Va.registerLoadRouter(e),Wa=e=>Va.getSaveHandlers(e),Ga=(e,t)=>Va.getLoadHandlers(e,t),Ka=`tensorflowjs`,qa=1,Ja=`models_store`,Ya=`model_info_store`;function Xa(){if(!O().getBool(`IS_BROWSER`))throw Error(`Failed to obtain IndexedDB factory because the current environmentis not a web browser.`);let e=typeof window>`u`?self:window,t=e.indexedDB||e.mozIndexedDB||e.webkitIndexedDB||e.msIndexedDB||e.shimIndexedDB;if(t==null)throw Error(`The current browser does not appear to support IndexedDB.`);return t}function Za(e){let t=e.result;t.createObjectStore(Ja,{keyPath:`modelPath`}),t.createObjectStore(Ya,{keyPath:`modelPath`})}var Qa=class{constructor(e){if(this.indexedDB=Xa(),e==null||!e)throw Error(`For IndexedDB, modelPath must not be null, undefined or empty.`);this.modelPath=e}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw Error(`BrowserLocalStorage.save() does not support saving model topology in binary formats yet.`);return this.databaseAction(this.modelPath,e)}async load(){return this.databaseAction(this.modelPath)}databaseAction(e,t){return new Promise((e,n)=>{let r=this.indexedDB.open(Ka,qa);r.onupgradeneeded=()=>Za(r),r.onsuccess=()=>{let i=r.result;if(t==null){let t=i.transaction(Ja,`readonly`),r=t.objectStore(Ja).get(this.modelPath);r.onsuccess=()=>{if(r.result==null)return i.close(),n(Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));e(r.result.modelArtifacts)},r.onerror=e=>(i.close(),n(r.error)),t.oncomplete=()=>i.close()}else{t.weightData=ua.join(t.weightData);let r=Fa(t),a=i.transaction(Ya,`readwrite`),o=a.objectStore(Ya),s;try{s=o.put({modelPath:this.modelPath,modelArtifactsInfo:r})}catch(e){return n(e)}let c;s.onsuccess=()=>{c=i.transaction(Ja,`readwrite`);let s=c.objectStore(Ja),l;try{l=s.put({modelPath:this.modelPath,modelArtifacts:t,modelArtifactsInfo:r})}catch(e){return n(e)}l.onsuccess=()=>e({modelArtifactsInfo:r}),l.onerror=e=>{o=a.objectStore(Ya);let t=o.delete(this.modelPath);t.onsuccess=()=>(i.close(),n(l.error)),t.onerror=e=>(i.close(),n(l.error))}},s.onerror=e=>(i.close(),n(s.error)),a.oncomplete=()=>{c==null?i.close():c.oncomplete=()=>i.close()}}},r.onerror=e=>n(r.error)})}};Qa.URL_SCHEME=`indexeddb://`;var $a=e=>O().getBool(`IS_BROWSER`)&&!Array.isArray(e)&&e.startsWith(Qa.URL_SCHEME)?eo(e.slice(Qa.URL_SCHEME.length)):null;Va.registerSaveRouter($a),Va.registerLoadRouter($a);function eo(e){return new Qa(e)}function to(e){return e.startsWith(Qa.URL_SCHEME)?e.slice(Qa.URL_SCHEME.length):e}var no=class{constructor(){this.indexedDB=Xa()}async listModels(){return new Promise((e,t)=>{let n=this.indexedDB.open(Ka,qa);n.onupgradeneeded=()=>Za(n),n.onsuccess=()=>{let r=n.result,i=r.transaction(Ya,`readonly`),a=i.objectStore(Ya).getAll();a.onsuccess=()=>{let t={};for(let e of a.result)t[e.modelPath]=e.modelArtifactsInfo;e(t)},a.onerror=e=>(r.close(),t(a.error)),i.oncomplete=()=>r.close()},n.onerror=e=>t(n.error)})}async removeModel(e){return e=to(e),new Promise((t,n)=>{let r=this.indexedDB.open(Ka,qa);r.onupgradeneeded=()=>Za(r),r.onsuccess=()=>{let i=r.result,a=i.transaction(Ya,`readwrite`),o=a.objectStore(Ya),s=o.get(e),c;s.onsuccess=()=>{if(s.result==null)return i.close(),n(Error(`Cannot find model with path '${e}' in IndexedDB.`));{let r=o.delete(e),a=()=>{c=i.transaction(Ja,`readwrite`);let r=c.objectStore(Ja).delete(e);r.onsuccess=()=>t(s.result.modelArtifactsInfo),r.onerror=e=>n(s.error)};r.onsuccess=a,r.onerror=e=>(a(),i.close(),n(s.error))}},s.onerror=e=>(i.close(),n(s.error)),a.oncomplete=()=>{c==null?i.close():c.oncomplete=()=>i.close()}},r.onerror=e=>n(r.error)})}},ro=`/`,io=`tensorflowjs_models`,ao=`info`,oo=`model_topology`,so=`weight_specs`,co=`weight_data`,lo=`model_metadata`;function uo(e){return{info:[io,e,ao].join(ro),topology:[io,e,oo].join(ro),weightSpecs:[io,e,so].join(ro),weightData:[io,e,co].join(ro),modelMetadata:[io,e,lo].join(ro)}}function fo(e){for(let t of Object.values(e))window.localStorage.removeItem(t)}function po(e){let t=e.split(ro);if(t.length<3)throw Error(`Invalid key format: ${e}`);return t.slice(1,t.length-1).join(ro)}function mo(e){return e.startsWith(ho.URL_SCHEME)?e.slice(ho.URL_SCHEME.length):e}var ho=class{constructor(e){if(!O().getBool(`IS_BROWSER`)||typeof window>`u`||window.localStorage===void 0)throw Error(`The current environment does not support local storage.`);if(this.LS=window.localStorage,e==null||!e)throw Error(`For local storage, modelPath must not be null, undefined or empty.`);this.modelPath=e,this.keys=uo(this.modelPath)}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw Error(`BrowserLocalStorage.save() does not support saving model topology in binary formats yet.`);{let t=JSON.stringify(e.modelTopology),n=JSON.stringify(e.weightSpecs),r=Fa(e),i=ua.join(e.weightData);try{this.LS.setItem(this.keys.info,JSON.stringify(r)),this.LS.setItem(this.keys.topology,t),this.LS.setItem(this.keys.weightSpecs,n),this.LS.setItem(this.keys.weightData,Oa(i));let a={format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,signature:e.signature==null?void 0:e.signature,userDefinedMetadata:e.userDefinedMetadata==null?void 0:e.userDefinedMetadata,modelInitializer:e.modelInitializer==null?void 0:e.modelInitializer,initializerSignature:e.initializerSignature==null?void 0:e.initializerSignature,trainingConfig:e.trainingConfig==null?void 0:e.trainingConfig};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(a)),{modelArtifactsInfo:r}}catch{throw fo(this.keys),Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${r.modelTopologyBytes}, weightSpecsBytes=${r.weightSpecsBytes}, weightDataBytes=${r.weightDataBytes}.`)}}}async load(){let e=JSON.parse(this.LS.getItem(this.keys.info));if(e==null)throw Error(`In local storage, there is no model with name '${this.modelPath}'`);if(e.modelTopologyType!==`JSON`)throw Error(`BrowserLocalStorage does not support loading non-JSON model topology yet.`);let t={},n=JSON.parse(this.LS.getItem(this.keys.topology));if(n==null)throw Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);t.modelTopology=n;let r=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(r==null)throw Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);t.weightSpecs=r;let i=this.LS.getItem(this.keys.modelMetadata);if(i!=null){let e=JSON.parse(i);t.format=e.format,t.generatedBy=e.generatedBy,t.convertedBy=e.convertedBy,e.signature!=null&&(t.signature=e.signature),e.userDefinedMetadata!=null&&(t.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(t.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(t.initializerSignature=e.initializerSignature),e.trainingConfig!=null&&(t.trainingConfig=e.trainingConfig)}let a=this.LS.getItem(this.keys.weightData);if(a==null)throw Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return t.weightData=ka(a),t}};ho.URL_SCHEME=`localstorage://`;var go=e=>O().getBool(`IS_BROWSER`)&&!Array.isArray(e)&&e.startsWith(ho.URL_SCHEME)?_o(e.slice(ho.URL_SCHEME.length)):null;Va.registerSaveRouter(go),Va.registerLoadRouter(go);function _o(e){return new ho(e)}var vo=class{constructor(){y(O().getBool(`IS_BROWSER`),()=>`Current environment is not a web browser`),y(typeof window>`u`||window.localStorage!==void 0,()=>`Current browser does not appear to support localStorage`),this.LS=window.localStorage}async listModels(){let e={},t=io+ro,n=ro+ao;for(let r=0;r<this.LS.length;++r){let i=this.LS.key(r);if(i.startsWith(t)&&i.endsWith(n)){let t=po(i);e[t]=JSON.parse(this.LS.getItem(i))}}return e}async removeModel(e){e=mo(e);let t=uo(e);if(this.LS.getItem(t.info)==null)throw Error(`Cannot find model at path '${e}'`);let n=JSON.parse(this.LS.getItem(t.info));return fo(t),n}},yo=`://`,bo=class e{constructor(){this.managers={}}static getInstance(){return e.instance??=new e,e.instance}static registerManager(t,n){y(t!=null,()=>`scheme must not be undefined or null.`),t.endsWith(yo)&&(t=t.slice(0,t.indexOf(yo))),y(t.length>0,()=>`scheme must not be an empty string.`);let r=e.getInstance();y(r.managers[t]==null,()=>`A model store manager is already registered for scheme '${t}'.`),r.managers[t]=n}static getManager(t){let n=e.getInstance().managers[t];if(n==null)throw Error(`Cannot find model manager for scheme '${t}'`);return n}static getSchemes(){return Object.keys(e.getInstance().managers)}};function xo(e){if(e.indexOf(yo)===-1)throw Error(`The url string provided does not contain a scheme. Supported schemes are: ${bo.getSchemes().join(`,`)}`);return{scheme:e.split(yo)[0],path:e.split(yo)[1]}}async function So(e,t,n=!1){y(e!==t,()=>`Old path and new path are the same: '${e}'`);let r=Va.getLoadHandlers(e);y(r.length>0,()=>`Copying failed because no load handler is found for source URL ${e}.`),y(r.length<2,()=>`Copying failed because more than one (${r.length}) load handlers for source URL ${e}.`);let i=r[0],a=Va.getSaveHandlers(t);y(a.length>0,()=>`Copying failed because no save handler is found for destination URL ${t}.`),y(a.length<2,()=>`Copying failed because more than one (${r.length}) save handlers for destination URL ${t}.`);let o=a[0],s=xo(e).scheme,c=xo(e).path,l=s===xo(e).scheme,u=await i.load();n&&l&&await bo.getManager(s).removeModel(c);let d=await o.save(u);return n&&!l&&await bo.getManager(s).removeModel(c),d.modelArtifactsInfo}async function Co(){let e=bo.getSchemes(),t={};for(let n of e){let e=await bo.getManager(n).listModels();for(let r in e){let i=n+yo+r;t[i]=e[r]}}return t}async function wo(e){let t=xo(e);return bo.getManager(t.scheme).removeModel(t.path)}async function To(e,t){return So(e,t,!1)}async function Eo(e,t){return So(e,t,!0)}var Do=class{constructor(){this.messageName=`setTimeoutCustom`,this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(e,t){return fetch(e,t)}now(){return performance.now()}encode(e,t){if(t!==`utf-8`&&t!==`utf8`)throw Error(`Browser's encoder only supports utf-8, but got ${t}`);return this.textEncoder??=new TextEncoder,this.textEncoder.encode(e)}decode(e,t){return new TextDecoder(t).decode(e)}setTimeoutCustom(e,t){if(typeof window>`u`||!O().getBool(`USE_SETTIMEOUTCUSTOM`)){setTimeout(e,t);return}this.functionRefs.push(e),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},`*`)},t),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener(`message`,e=>{if(e.source===window&&e.data.name===this.messageName){e.stopPropagation();let t=this.functionRefs[e.data.index];t(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}isTypedArray(e){return Fr(e)}};if(O().get(`IS_BROWSER`)){O().setPlatform(`browser`,new Do);try{bo.registerManager(ho.URL_SCHEME,new vo)}catch{}try{bo.registerManager(Qa.URL_SCHEME,new no)}catch{}}var Oo=o(((e,t)=>{t.exports={}})),ko={importFetch:()=>Oo()},Ao,jo=class{constructor(){this.util=Oo(),this.textEncoder=new this.util.TextEncoder}fetch(e,t){return O().global.fetch==null?(Ao??=ko.importFetch(),Ao(e,t)):O().global.fetch(e,t)}now(){let e=process.hrtime();return e[0]*1e3+e[1]/1e6}encode(e,t){if(t!==`utf-8`&&t!==`utf8`)throw Error(`Node built-in encoder only supports utf-8, but got ${t}`);return this.textEncoder.encode(e)}decode(e,t){return e.length===0?``:new this.util.TextDecoder(t).decode(e)}isTypedArray(e){return this.util.types.isFloat32Array(e)||this.util.types.isInt32Array(e)||this.util.types.isUint8Array(e)||this.util.types.isUint8ClampedArray(e)}};O().get(`IS_NODE`)&&!O().get(`IS_BROWSER`)&&O().setPlatform(`node`,new jo);function I(e,t=`float32`,n){return t||=`float32`,Se(e),new xi(e,t,n)}function Mo(e,t){let n=M(e,`x`,`cast`);if(!se(t))throw Error(`Failed to cast to unknown dtype ${t}`);if(t===`string`&&n.dtype!==`string`||t!==`string`&&n.dtype===`string`)throw Error(`Only strings can be casted to strings`);let r={x:n},i={dtype:t};return j.runKernel(nt,r,i)}var L=N({cast_:Mo});function No(e){let t={x:M(e,`x`,`clone`,`string_or_numeric`)};return j.runKernel(Bt,t)}var Po=N({clone_:No});function Fo(e,t=!1){console.log(e.toString(t))}qi(),Ti({buffer:I,cast:L,clone:Po,print:Fo});function Io(e,t){let n=M(e,`a`,`add`),r=M(t,`b`,`add`);[n,r]=A(n,r);let i={a:n,b:r};return j.runKernel(`Add`,i)}var R=N({add_:Io});function Lo(e,t){let n=M(e,`a`,`floorDiv`),r=M(t,`b`,`floorDiv`);[n,r]=A(n,r);let i={a:n,b:r};return j.runKernel(Pt,i)}var Ro=N({floorDiv_:Lo});function zo(e,t){let n=M(e,`a`,`div`),r=M(t,`b`,`div`);if([n,r]=A(n,r),n.dtype===`int32`&&r.dtype===`int32`)return Ro(n,r);let i={a:n,b:r};return j.runKernel(Tt,i,{})}var z=N({div_:zo});function Bo(e,t){let n=M(e,`a`,`mul`),r=M(t,`b`,`mul`);[n,r]=A(n,r);let i={a:n,b:r};return j.runKernel(fn,i)}var B=N({mul_:Bo});function Vo(e){let t=M(e,`x`,`abs`);if(t.dtype===`complex64`){let e={x:t};return j.runKernel(ot,e)}else{let e={x:t};return j.runKernel(`Abs`,e)}}var Ho=N({abs_:Vo});function Uo(e){let t={x:M(e,`x`,`acos`)};return j.runKernel(Le,t)}var Wo=N({acos_:Uo});function Go(e){let t={x:M(e,`x`,`acosh`)};return j.runKernel(Re,t)}var Ko=N({acosh_:Go});function qo(e){y(Array.isArray(e),()=>`The argument passed to tf.addN() must be a list of tensors`),y(e.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${e.length}`);let t=e.map((e,t)=>M(e,`tensors${t}`,`addN`)),n=t[0];t.forEach(e=>{if(e.dtype!==n.dtype)throw Error(`All tensors passed to tf.addN() must have the same dtype`)}),t.forEach(e=>{if(!w(e.shape,n.shape))throw Error(`All tensors passed to tf.addN() must have the same shape`)});let r=t;return j.runKernel(ze,r)}var Jo=N({addN_:qo});function Yo(e,t=null,n=!1){let r={x:M(e,`x`,`all`,`bool`)},i={axis:t,keepDims:n};return j.runKernel(`All`,r,i)}var Xo=N({all_:Yo});function Zo(e,t=null,n=!1){let r={x:M(e,`x`,`any`,`bool`)},i={axis:t,keepDims:n};return j.runKernel(`Any`,r,i)}var Qo=N({any_:Zo});function $o(e,t=0){let n={x:M(e,`x`,`argMax`)},r={axis:t};return j.runKernel(Be,n,r)}var es=N({argMax_:$o});function ts(e,t=0){let n={x:M(e,`x`,`argMin`)},r={axis:t};return j.runKernel(Ve,n,r)}var ns=N({argMin_:ts});function rs(e){let t={x:M(e,`x`,`asin`)};return j.runKernel(He,t)}var is=N({asin_:rs});function as(e){let t={x:M(e,`x`,`asinh`)};return j.runKernel(Ue,t)}var os=N({asinh_:as});function ss(e){let t={x:M(e,`x`,`atan`)};return j.runKernel(We,t)}var cs=N({atan_:ss});function ls(e,t){let n=M(e,`a`,`atan2`),r=M(t,`b`,`atan2`);[n,r]=A(n,r);let i={a:n,b:r};return j.runKernel(Ke,i)}var us=N({atan2_:ls});function ds(e){let t={x:M(e,`x`,`atanh`)};return j.runKernel(Ge,t)}var fs=N({atanh_:ds});function ps(e,t,n,r,i=`NHWC`,a){let o=e[3];return gs(e,[...t,o],n,a,r,null,null,As(i))}function ms(e,t,n,r,i,a,o=`channelsLast`){let[s,c]=xs(t),l;if(o===`channelsLast`)l=[s,c,e[3],e[3]];else if(o===`channelsFirst`)l=[s,c,e[1],e[1]];else throw Error(`Unknown dataFormat ${o}`);return gs(e,l,n,r,i,a,!1,o)}function hs(e,t,n,r,i,a,o=`NDHWC`){let[s,c,l]=Ss(t),u,d;if(o===`NDHWC`)d=`channelsLast`,u=[s,c,l,e[4],e[4]];else if(o===`NCDHW`)d=`channelsFirst`,u=[s,c,l,e[1],e[1]];else throw Error(`Unknown dataFormat ${o}`);return _s(e,u,n,r,i,!1,d,a)}function gs(e,t,n,r,i,a,o=!1,s=`channelsLast`){let[c,l,u,d]=[-1,-1,-1,-1];if(s===`channelsLast`)[c,l,u,d]=e;else if(s===`channelsFirst`)[c,d,l,u]=e;else throw Error(`Unknown dataFormat ${s}`);let[f,p,,m]=t,[h,g]=xs(n),[_,v]=xs(r),y=Cs(f,_),b=Cs(p,v),{padInfo:x,outHeight:S,outWidth:C}=ws(i,l,u,h,g,y,b,a,s),w=o?m*d:m,ee;return s===`channelsFirst`?ee=[c,w,S,C]:s===`channelsLast`&&(ee=[c,S,C,w]),{batchSize:c,dataFormat:s,inHeight:l,inWidth:u,inChannels:d,outHeight:S,outWidth:C,outChannels:w,padInfo:x,strideHeight:h,strideWidth:g,filterHeight:f,filterWidth:p,effectiveFilterHeight:y,effectiveFilterWidth:b,dilationHeight:_,dilationWidth:v,inShape:e,outShape:ee,filterShape:t}}function _s(e,t,n,r,i,a=!1,o=`channelsLast`,s){let[c,l,u,d,f]=[-1,-1,-1,-1,-1];if(o===`channelsLast`)[c,l,u,d,f]=e;else if(o===`channelsFirst`)[c,f,l,u,d]=e;else throw Error(`Unknown dataFormat ${o}`);let[p,m,h,,g]=t,[_,v,y]=Ss(n),[b,x,S]=Ss(r),C=Cs(p,b),w=Cs(m,x),ee=Cs(h,S),{padInfo:T,outDepth:E,outHeight:te,outWidth:ne}=Ts(i,l,u,d,_,v,y,C,w,ee,s),D=a?g*f:g,re;return o===`channelsFirst`?re=[c,D,E,te,ne]:o===`channelsLast`&&(re=[c,E,te,ne,D]),{batchSize:c,dataFormat:o,inDepth:l,inHeight:u,inWidth:d,inChannels:f,outDepth:E,outHeight:te,outWidth:ne,outChannels:D,padInfo:T,strideDepth:_,strideHeight:v,strideWidth:y,filterDepth:p,filterHeight:m,filterWidth:h,effectiveFilterDepth:C,effectiveFilterHeight:w,effectiveFilterWidth:ee,dilationDepth:b,dilationHeight:x,dilationWidth:S,inShape:e,outShape:re,filterShape:t}}function vs(e,t,n,r,i){r??=bs(e,t,n);let a=e[0],o=e[1];return[Es((a-t+2*r)/n+1,i),Es((o-t+2*r)/n+1,i)]}function ys(e,t,n,r,i,a){i??=bs(e,t[0],r[0]);let o=[0,0,0,n];for(let n=0;n<3;n++)e[n]+2*i>=t[n]&&(o[n]=Es((e[n]-t[n]+2*i)/r[n]+1,a));return o}function bs(e,t,n,r=1){let i=Cs(t,r);return Math.floor((e[0]*(n-1)-n+i)/2)}function xs(e){return typeof e==`number`?[e,e,e]:e.length===2?[e[0],e[1],1]:e}function Ss(e){return typeof e==`number`?[e,e,e]:e}function Cs(e,t){return t<=1?e:e+(e-1)*(t-1)}function ws(e,t,n,r,i,a,o,s,c){let l,u,d;if(typeof e==`number`){l={top:e,bottom:e,left:e,right:e,type:e===0?`VALID`:`NUMBER`};let i=vs([t,n],a,r,e,s);u=i[0],d=i[1]}else if(e===`same`){u=Math.ceil(t/r),d=Math.ceil(n/i);let e=Math.max(0,(u-1)*r+a-t),s=Math.max(0,(d-1)*i+o-n),c=Math.floor(e/2),f=e-c,p=Math.floor(s/2);l={top:c,bottom:f,left:p,right:s-p,type:`SAME`}}else if(e===`valid`)l={top:0,bottom:0,left:0,right:0,type:`VALID`},u=Math.ceil((t-a+1)/r),d=Math.ceil((n-o+1)/i);else if(typeof e==`object`){let f=c===`channelsLast`?e[1][0]:e[2][0],p=c===`channelsLast`?e[1][1]:e[2][1],m=c===`channelsLast`?e[2][0]:e[3][0],h=c===`channelsLast`?e[2][1]:e[3][1];l={top:f,bottom:p,left:m,right:h,type:f===0&&p===0&&m===0&&h===0?`VALID`:`EXPLICIT`},u=Es((t-a+f+p)/r+1,s),d=Es((n-o+m+h)/i+1,s)}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:l,outHeight:u,outWidth:d}}function Ts(e,t,n,r,i,a,o,s,c,l,u){let d,f,p,m;if(e===`valid`&&(e=0),typeof e==`number`){d={top:e,bottom:e,left:e,right:e,front:e,back:e,type:e===0?`VALID`:`NUMBER`};let h=ys([t,n,r,1],[s,c,l],1,[i,a,o],e,u);f=h[0],p=h[1],m=h[2]}else if(e===`same`){f=Math.ceil(t/i),p=Math.ceil(n/a),m=Math.ceil(r/o);let e=(f-1)*i+s-t,u=(p-1)*a+c-n,h=(m-1)*o+l-r,g=Math.floor(e/2),_=e-g,v=Math.floor(u/2),y=u-v,b=Math.floor(h/2);d={top:v,bottom:y,left:b,right:h-b,front:g,back:_,type:`SAME`}}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:d,outDepth:f,outHeight:p,outWidth:m}}function Es(e,t){if(!t)return Math.trunc(e);switch(t){case`round`:return Math.round(e);case`ceil`:return Math.ceil(e);case`floor`:return Math.floor(e);default:throw Error(`Unknown roundingMode ${t}`)}}function Ds(e){let[t,n,r]=xs(e);return t===1&&n===1&&r===1}function Os(e,t){return Ds(e)||Ds(t)}function ks(e){return xs(e).every(e=>e>0)}function As(e){if(e===`NHWC`)return`channelsLast`;if(e===`NCHW`)return`channelsFirst`;throw Error(`Unknown dataFormat ${e}`)}function js(e,t,n){if(n!=null){if(typeof t==`string`)throw Error(`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);if(typeof t==`number`)y(ee(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);else if(typeof t==`object`)t.forEach(t=>{t.forEach(t=>{y(ee(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`)})});else throw Error(`Error in ${e}: Unknown padding parameter: ${t}`)}}function Ms(e,t){let n={x:M(e,`x`,`reshape`,`string_or_numeric`)},r={shape:t};return j.runKernel(An,n,r)}var V=N({reshape_:Ms});function Ns(e,t,n,r,i){let a=M(e,`x`,`avgPool`,`float32`);y(Os(n,1),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${n} and dilations '1'`);let o=a,s=!1;a.rank===3&&(s=!0,o=V(a,[1,a.shape[0],a.shape[1],a.shape[2]])),y(o.rank===4,()=>`Error in avgPool: x must be rank 4 but got rank ${o.rank}.`),js(`avgPool`,r,i);let c={x:o},l={filterSize:t,strides:n,pad:r,dimRoundingMode:i},u=j.runKernel(qe,c,l);return u=L(u,a.dtype),s?V(u,[u.shape[1],u.shape[2],u.shape[3]]):u}var Ps=N({avgPool_:Ns});function Fs(e,t,n,r,i,a=`NDHWC`){let o=M(e,`x`,`avgPool3d`,`float32`),s=o,c=!1;o.rank===4&&(c=!0,s=V(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),y(s.rank===5,()=>`Error in avgPool3d: x must be rank 5 but got rank ${s.rank}.`),y(a===`NDHWC`,()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),y(typeof n==`number`&&n>0||Array.isArray(n)&&n[0]>0&&n[1]>0&&n[2]>0,()=>`Error in avgPool3d: Stride must be > 0, but got '${n}'`),js(`avgPool3d`,r,i);let l={x:s},u={filterSize:t,strides:n,pad:r,dimRoundingMode:i,dataFormat:a},d=j.runKernel(Ye,l,u);return d=L(d,s.dtype),c?V(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}var Is=N({avgPool3d_:Fs});function Ls(e,t=0){y(e.length>=1,()=>`Pass at least one tensor to concat`);let n=ra(e,`tensors`,`concat`,`string_or_numeric`);if(n[0].dtype===`complex64`&&n.forEach(e=>{if(e.dtype!==`complex64`)throw Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${e.dtype}. `)}),n.length===1)return Po(n[0]);let r=n,i={axis:t};return j.runKernel(st,r,i)}var Rs=N({concat_:Ls});function zs(e,t,n=!1,r=!1){let i=M(e,`a`,`matMul`),a=M(t,`b`,`matMul`);[i,a]=A(i,a);let o={a:i,b:a},s={transposeA:n,transposeB:r};return j.runKernel(Ze,o,s)}var H=N({matMul_:zs});function Bs(e){let t={x:M(e,`x`,`sigmoid`,`float32`)};return j.runKernel(qn,t)}var Vs=N({sigmoid_:Bs});function Hs(e,t,n){let r=M(e,`x`,`slice`,`string_or_numeric`);if(r.rank===0)throw Error(`Slicing scalar is not possible`);let i={x:r},a={begin:t,size:n};return j.runKernel(Wn,i,a)}var U=N({slice_:Hs});function Us(e){let t={x:M(e,`x`,`tanh`,`float32`)};return j.runKernel(dr,t)}var Ws=N({tanh_:Us});function Gs(e,t,n,r,i,a){let o=M(e,`forgetBias`,`basicLSTMCell`),s=M(t,`lstmKernel`,`basicLSTMCell`),c=M(n,`lstmBias`,`basicLSTMCell`),l=M(r,`data`,`basicLSTMCell`),u=M(i,`c`,`basicLSTMCell`),d=R(H(Rs([l,M(a,`h`,`basicLSTMCell`)],1),s),c),f=d.shape[0],p=d.shape[1]/4,m=[f,p],h=U(d,[0,0],m),g=U(d,[0,p],m),_=U(d,[0,p*2],m),v=U(d,[0,p*3],m),y=R(B(Vs(h),Ws(g)),B(u,Vs(R(o,_))));return[y,B(Ws(y),Vs(v))]}var Ks=N({basicLSTMCell_:Gs});function qs(e,t,n){let r=M(e,`x`,`batchToSpaceND`),i=t.reduce((e,t)=>e*t);y(r.rank>=1+t.length,()=>`input rank is ${r.rank} but should be > than blockShape.length ${t.length}`),y(n.length===t.length,()=>`crops.length is ${n.length} but should be equal to blockShape.length  ${t.length}`),y(r.shape[0]%i===0,()=>`input tensor batch is ${r.shape[0]} but is not divisible by the product of the elements of blockShape ${t.join(` * `)} === ${i}`);let a={x:r},o={blockShape:t,crops:n};return j.runKernel(Qe,a,o)}var Js=N({batchToSpaceND_:qs});function Ys(e){let t;return t=e.rank===0||e.rank===1?V(e,[1,1,1,e.size]):e.rank===2?V(e,[1,1,e.shape[0],e.shape[1]]):e.rank===3?V(e,[1,e.shape[0],e.shape[1],e.shape[2]]):e,t}function Xs(e,t,n,r,i,a){a??=.001;let o=M(e,`x`,`batchNorm`),s=M(t,`mean`,`batchNorm`),c=M(n,`variance`,`batchNorm`),l;i!=null&&(l=M(i,`scale`,`batchNorm`));let u;r!=null&&(u=M(r,`offset`,`batchNorm`)),y(s.rank===c.rank,()=>`Batch normalization gradient requires mean and variance to have equal ranks.`),y(u==null||s.rank===u.rank,()=>`Batch normalization gradient requires mean and offset to have equal ranks.`),y(l==null||s.rank===l.rank,()=>`Batch normalization gradient requires mean and scale to have equal ranks.`);let d={x:Ys(o),scale:l,offset:u,mean:s,variance:c},f={varianceEpsilon:a};return V(j.runKernel(Ft,d,f),o.shape)}var Zs=N({batchNorm_:Xs});function Qs(e,t,n,r,i,a){let o=M(e,`x`,`batchNorm`),s=M(t,`mean`,`batchNorm`),c=M(n,`variance`,`batchNorm`),l;i!=null&&(l=M(i,`scale`,`batchNorm`));let u;return r!=null&&(u=M(r,`offset`,`batchNorm`)),y(o.rank===2,()=>`Error in batchNorm2D: x must be rank 2 but got rank ${o.rank}.`),y(s.rank===2||s.rank===1,()=>`Error in batchNorm2D: mean must be rank 2 or rank 1 but got rank ${s.rank}.`),y(c.rank===2||c.rank===1,()=>`Error in batchNorm2D: variance must be rank 2 or rank 1 but got rank ${c.rank}.`),l!=null&&y(l.rank===2||l.rank===1,()=>`Error in batchNorm2D: scale must be rank 2 or rank 1 but got rank ${l.rank}.`),u!=null&&y(u.rank===2||u.rank===1,()=>`Error in batchNorm2D: offset must be rank 2 or rank 1 but got rank ${u.rank}.`),Zs(o,s,c,u,l,a)}var $s=N({batchNorm2d_:Qs});function ec(e,t,n,r,i,a){let o=M(e,`x`,`batchNorm`),s=M(t,`mean`,`batchNorm`),c=M(n,`variance`,`batchNorm`),l;i!=null&&(l=M(i,`scale`,`batchNorm`));let u;return r!=null&&(u=M(r,`offset`,`batchNorm`)),y(o.rank===3,()=>`Error in batchNorm3D: x must be rank 3 but got rank ${o.rank}.`),y(s.rank===3||s.rank===1,()=>`Error in batchNorm3D: mean must be rank 3 or rank 1 but got rank ${s.rank}.`),y(c.rank===3||c.rank===1,()=>`Error in batchNorm3D: variance must be rank 3 or rank 1 but got rank ${c.rank}.`),l!=null&&y(l.rank===3||l.rank===1,()=>`Error in batchNorm3D: scale must be rank 3 or rank 1 but got rank ${l.rank}.`),u!=null&&y(u.rank===3||u.rank===1,()=>`Error in batchNorm3D: offset must be rank 3 or rank 1 but got rank ${u.rank}.`),Zs(o,s,c,u,l,a)}var tc=N({batchNorm3d_:ec});function nc(e,t,n,r,i,a){let o=M(e,`x`,`batchNorm`),s=M(t,`mean`,`batchNorm`),c=M(n,`variance`,`batchNorm`),l;i!=null&&(l=M(i,`scale`,`batchNorm`));let u;return r!=null&&(u=M(r,`offset`,`batchNorm`)),y(o.rank===4,()=>`Error in batchNorm4D: x must be rank 4 but got rank ${o.rank}.`),y(s.rank===4||s.rank===1,()=>`Error in batchNorm4D: mean must be rank 4 or rank 1 but got rank ${s.rank}.`),y(c.rank===4||c.rank===1,()=>`Error in batchNorm4D: variance must be rank 4 or rank 1 but got rank ${c.rank}.`),l!=null&&y(l.rank===4||l.rank===1,()=>`Error in batchNorm4D: scale must be rank 4 or rank 1 but got rank ${l.rank}.`),u!=null&&y(u.rank===4||u.rank===1,()=>`Error in batchNorm4D: offset must be rank 4 or rank 1 but got rank ${u.rank}.`),Zs(o,s,c,u,l,a)}var rc=N({batchNorm4d_:nc});function ic(e,t,n){let r=M(e,`x`,`bincount`),i=M(t,`weights`,`bincount`);y(r.dtype===`int32`,()=>`Error in bincount: input dtype must be int32, but got ${r.dtype}`),y(n>=0,()=>`size must be non-negative, but got ${n}.`),y(i.size===r.size||i.size===0,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${r.shape}, weights shape: ${i.shape}.`);let a={x:r,weights:i},o={size:n};return j.runKernel($e,a,o)}var ac=N({bincount_:ic});function oc(e,t){let n=M(e,`x`,`bitwiseAnd`),r=M(t,`y`,`bitwiseAnd`);if(!w(n.shape,r.shape))throw Error(`BitwiseAnd: Tensors must have the same shape. x: ${n.shape}, y: ${r.shape}`);if(n.dtype!==`int32`||r.dtype!==`int32`)throw Error(`BitwiseAnd: Only supports 'int32' values in tensor, found type of x: ${n.dtype} and type of y: ${r.dtype}`);let i={a:n,b:r};return j.runKernel(et,i)}var sc=N({bitwiseAnd_:oc});function cc(e,t){let n=M(e,`s0`,`broadcastArgs`,`int32`),r=M(t,`s1`,`broadcastArgs`,`int32`);if(n.rank!==1)throw Error(`broadcastArgs(): first input must be a vector (rank=1). Has rank ${n.rank}`);if(r.rank!==1)throw Error(`broadcastArgs(): second input must be a vector (rank=1). Has rank ${r.rank}`);let i={s0:n,s1:r};return j.runKernel(tt,i)}var lc=N({broadcastArgs_:cc});function uc(e,t){let n=M(e,`broadcastTo`,`x`),r=n.shape;if(Se(t),t.length<n.rank)throw Error(`broadcastTo(): shape.length=${t.length} < input.rank=${n.rank}.`);if(t.length>n.rank){let e=n.shape.slice();for(;e.length<t.length;)e.unshift(1);n=V(n,e)}let i=n.shape,a=Array.from(t);for(let e=t.length-1;e>=0;e--)if(i[e]===t[e])a[e]=1;else if(n.shape[e]!==1)throw Error(`broadcastTo(): [${r}] cannot be broadcast to [${t}].`);if(a.map((e,t)=>e>1?t:-1).filter(e=>e>=0).length===0)return Po(n);let o={x:n},s={reps:a};return j.runKernel(fr,o,s)}var dc=N({broadcastTo_:uc});function fc(e){let t={x:M(e,`x`,`ceil`,`float32`)};return j.runKernel(rt,t)}var pc=N({ceil_:fc});function mc(e,t,n){Se(e),n||=me(t);let r={shape:e,value:t,dtype:n};return j.runKernel(jt,{},r)}function hc(e,t,n){let r=M(e,`x`,`clipByValue`);if(y(t<=n,()=>`Error in clip: min (${t}) must be less than or equal to max (${n}).`),t===n)return mc(r.shape,t,r.dtype);let i={x:r},a={clipValueMin:t,clipValueMax:n};return j.runKernel(it,i,a)}var gc=N({clipByValue_:hc});function _c(e){return Rs(e,0)}var vc=N({concat1d_:_c});function yc(e,t){return Rs(e,t)}var bc=N({concat2d_:yc});function xc(e,t){return Rs(e,t)}var Sc=N({concat3d_:xc});function Cc(e,t){return Rs(e,t)}var wc=N({concat4d_:Cc});function Tc(e,t,n,r,i=`NHWC`,a=[1,1],o){let s=M(e,`x`,`conv2d`,`float32`),c=M(t,`filter`,`conv2d`,`float32`),l=s,u=!1;s.rank===3&&(u=!0,l=V(s,[1,s.shape[0],s.shape[1],s.shape[2]])),y(l.rank===4,()=>`Error in conv2d: input must be rank 4, but got rank ${l.rank}.`),y(c.rank===4,()=>`Error in conv2d: filter must be rank 4, but got rank ${c.rank}.`),js(`conv2d`,r,o);let d=i===`NHWC`?l.shape[3]:l.shape[1];y(d===c.shape[2],()=>`Error in conv2d: depth of input (${d}) must match input depth for filter ${c.shape[2]}.`),y(Os(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),y(ks(a),()=>`Error in conv2D: Dilated rates should be larger than 0.`),y(ks(n),()=>`Error in conv2D: Strides should be larger than 0.`);let f={x:l,filter:c},p={strides:n,pad:r,dataFormat:i,dilations:a,dimRoundingMode:o},m=j.runKernel(ct,f,p);return u?V(m,[m.shape[1],m.shape[2],m.shape[3]]):m}var Ec=N({conv2d_:Tc});function Dc(e,t,n,r,i=`NWC`,a=1,o){let s=M(e,`x`,`conv1d`),c=M(t,`filter`,`conv1d`),l=s,u=!1;s.rank===2&&(u=!0,l=V(s,[1,s.shape[0],s.shape[1]])),y(l.rank===3,()=>`Error in conv1d: input must be rank 3, but got rank ${l.rank}.`),y(c.rank===3,()=>`Error in conv1d: filter must be rank 3, but got rank ${c.rank}.`),js(`conv1d`,r,o),y(l.shape[2]===c.shape[1],()=>`Error in conv1d: depth of input (${l.shape[2]}) must match input depth for filter ${c.shape[1]}.`),y(Os(n,a),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${n} and dilation '${a}'`),y(ks(a),()=>`Error in conv1D: Dilated rates should be larger than 0.`),y(ks(n),()=>`Error in conv1D: Stride should be larger than 0.`),y(i===`NWC`,()=>`Error in conv1d: got dataFormat of ${i} but only NWC is currently supported.`);let d=V(c,[1,c.shape[0],c.shape[1],c.shape[2]]),f=Ec(V(l,[l.shape[0],1,l.shape[1],l.shape[2]]),d,[1,n],r,`NHWC`,[1,a],o);return u?V(f,[f.shape[2],f.shape[3]]):V(f,[f.shape[0],f.shape[2],f.shape[3]])}var Oc=N({conv1d_:Dc});function kc(e,t,n,r,i,a=`NHWC`,o){y(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let s=e,c=t,l=!1;t.rank===3&&(l=!0,c=V(t,[1,t.shape[0],t.shape[1],t.shape[2]]),s=[1,e[0],e[1],e[2]]),y(s.length===4,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${s.length}.`),y(c.rank===4,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${c.rank}`),y(n.rank===4,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${n.rank}`);let u=a===`NHWC`?s[3]:s[1],d=a===`NHWC`?c.shape[3]:c.shape[1];y(u===n.shape[2],()=>`Error in conv2dDerInput: depth of input (${u}) must match input depth for filter ${n.shape[2]}.`),y(d===n.shape[3],()=>`Error in conv2dDerInput: depth of output (${d}) must match output depth for filter ${n.shape[3]}.`),js(`conv2dDerInput`,i,o);let f={dy:c,filter:n},p={strides:r,pad:i,dataFormat:a,dimRoundingMode:o,inputShape:s},m=j.runKernel(ut,f,p);return l?V(m,[m.shape[1],m.shape[2],m.shape[3]]):m}var Ac=N({conv2DBackpropInput_:kc});function jc(e,t,n,r,i,a){return Ac(n,M(e,`x`,`conv2dTranspose`),M(t,`filter`,`conv2dTranspose`),r,i,`NHWC`,a)}var Mc=N({conv2dTranspose_:jc});function Nc(e,t,n,r,i=`NDHWC`,a=[1,1,1]){let o=M(e,`x`,`conv3d`),s=M(t,`filter`,`conv3d`),c=o,l=!1;o.rank===4&&(l=!0,c=V(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),y(c.rank===5,()=>`Error in conv3d: input must be rank 5, but got rank ${c.rank}.`),y(s.rank===5,()=>`Error in conv3d: filter must be rank 5, but got rank ${s.rank}.`),y(c.shape[4]===s.shape[3],()=>`Error in conv3d: depth of input (${c.shape[4]}) must match input depth for filter ${s.shape[3]}.`),y(Os(n,a),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),y(i===`NDHWC`,()=>`Error in conv3d: got dataFormat of ${i} but only NDHWC is currently supported.`),y(ks(a),()=>`Error in conv3D: Dilated rates should be larger than 0.`),y(ks(n),()=>`Error in conv3D: Strides should be larger than 0.`);let u={x:c,filter:s},d={strides:n,pad:r,dataFormat:i,dilations:a},f=j.runKernel(dt,u,d);return l?V(f,[f.shape[1],f.shape[2],f.shape[3],f.shape[4]]):f}var Pc=N({conv3d_:Nc});function Fc(e,t,n,r,i){y(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let a=e,o=t,s=!1;t.rank===4&&(s=!0,o=V(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]]),a=[1,e[0],e[1],e[2],e[3]]);let c=a[4],l=o.shape[4];y(a.length===5,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${a.length}.`),y(o.rank===5,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${o.rank}`),y(n.rank===5,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${n.rank}`),y(c===n.shape[3],()=>`Error in conv3dDerInput: depth of input (${c}) must match input depth for filter ${n.shape[3]}.`),y(l===n.shape[4],()=>`Error in conv3dDerInput: depth of output (${l}) must match output depth for filter ${n.shape[4]}.`);let u={dy:o,filter:n},d={pad:i,strides:r,inputShape:a},f=j.runKernel(pt,u,d);return s?V(f,[f.shape[1],f.shape[2],f.shape[3],f.shape[4]]):f}var Ic=N({conv3DBackpropInput_:Fc});function Lc(e,t,n,r,i){return Ic(n,M(e,`x`,`conv3dTranspose`),M(t,`filter`,`conv3dTranspose`),r,i)}var Rc=N({conv3dTranspose_:Lc});function zc(e){let t={x:M(e,`x`,`cos`,`float32`)};return j.runKernel(`Cos`,t)}var Bc=N({cos_:zc});function Vc(e){let t={x:M(e,`x`,`cosh`,`float32`)};return j.runKernel(mt,t)}var Hc=N({cosh_:Vc});function Uc(e,t=0,n=!1,r=!1){let i={x:M(e,`x`,`cumprod`)},a={axis:t,exclusive:n,reverse:r};return j.runKernel(ht,i,a)}var Wc=N({cumprod_:Uc});function Gc(e,t=0,n=!1,r=!1){let i={x:M(e,`x`,`cumsum`)},a={axis:t,exclusive:n,reverse:r};return j.runKernel(gt,i,a)}var Kc=N({cumsum_:Gc});function qc(e,t,n,r=!1){let i=M(e,`x`,`denseBincount`),a=M(t,`weights`,`denseBincount`);y(i.dtype===`int32`,()=>`Error in denseBincount: input dtype must be int32, but got ${i.dtype}`),y(i.rank<=2,()=>`Error in denseBincount: input must be at most rank 2, but got rank ${i.rank}.`),y(n>=0,()=>`size must be non-negative, but got ${n}.`),y(a.size===i.size||a.size===0,()=>`Error in denseBincount: weights must have the same shape as x or 0-length, but got x shape: ${i.shape}, weights shape: ${a.shape}.`);let o={x:i,weights:a},s={size:n,binaryOutput:r};return j.runKernel(vt,o,s)}var Jc=N({denseBincount_:qc});function Yc(e,t,n=`NHWC`){let r=M(e,`x`,`depthToSpace`,`float32`),i=n===`NHWC`?r.shape[1]:r.shape[2],a=n===`NHWC`?r.shape[2]:r.shape[3],o=n===`NHWC`?r.shape[3]:r.shape[1];y(t>1,()=>`blockSize should be > 1 for depthToSpace, but was: ${t}`),y(i*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${i} and ${t}  for depthToSpace with input shape
    ${r.shape}`),y(a*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${a} and ${t} for depthToSpace with input shape
        ${r.shape}`),y(o%(t*t)===0,()=>`Dimension size must be evenly divisible by ${t*t} but is ${o} for depthToSpace with input shape ${r.shape}`);let s={x:r},c={blockSize:t,dataFormat:n};return j.runKernel(yt,s,c)}var Xc=N({depthToSpace_:Yc});function Zc(e,t,n,r,i=`NHWC`,a=[1,1],o){let s=M(e,`x`,`depthwiseConv2d`,`float32`),c=M(t,`filter`,`depthwiseConv2d`,`float32`),l=s,u=!1;s.rank===3&&(u=!0,l=V(s,[1,s.shape[0],s.shape[1],s.shape[2]])),y(l.rank===4,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${l.rank}.`),y(c.rank===4,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${c.rank}.`);let d=i===`NHWC`?l.shape[3]:l.shape[1];y(d===c.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${d}) must match the inChannels dimension in filter ${c.shape[2]}.`),js(`depthwiseConv2d`,r,o);let f={x:l,filter:c},p={strides:n,pad:r,dataFormat:i,dilations:a,dimRoundingMode:o},m=j.runKernel(bt,f,p);return u?V(m,[m.shape[1],m.shape[2],m.shape[3]]):m}var Qc=N({depthwiseConv2d_:Zc});function $c(e){let t={x:M(e,`x`,`diag`)};return j.runKernel(Ct,t)}var el=N({diag_:$c});function tl(e,t,n,r,i=[1,1],a=`NHWC`){let o=M(e,`x`,`dilation2d`),s=M(t,`filter`,`dilation2d`);y(o.rank===3||o.rank===4,()=>`Error in dilation2d: input must be rank 3 or 4, but got rank ${o.rank}.`),y(s.rank===3,()=>`Error in dilation2d: filter must be rank 3, but got rank ${s.rank}.`),y(a===`NHWC`,()=>`Error in dilation2d: Only NHWC is currently supported, but got dataFormat of ${a}`);let c=o,l=!1;o.rank===3&&(c=V(o,[1,o.shape[0],o.shape[1],o.shape[2]]),l=!0),y(c.shape[3]===s.shape[2],()=>`Error in dilation2d:  input and filter must have the same depth: ${c.shape[3]} vs ${s.shape[2]}`);let u={x:c,filter:s},d={strides:n,pad:r,dilations:i},f=j.runKernel(wt,u,d);return l?V(f,[f.shape[1],f.shape[2],f.shape[3]]):f}var nl=N({dilation2d_:tl});function rl(e,t){let n=e.length,r=[];for(let i=0;i<n;i++){let a=n-1-i,o=e[a]||1;(t[t.length-1-i]||1)>1&&o===1&&r.unshift(a)}return r}function il(e,t){let n=[];for(let r=0;r<t.length;r++){let i=e[e.length-r-1],a=t.length-r-1,o=t[a];(i==null||i===1&&o>1)&&n.unshift(a)}return n}function W(e,t){let n=Math.max(e.length,t.length),r=Array(n);for(let i=0;i<n;i++){let a=e[e.length-i-1];a??=1;let o=t[t.length-i-1];if(o??=1,a===1)r[n-i-1]=o;else if(o===1)r[n-i-1]=a;else if(a!==o){let n=`Operands could not be broadcast together with shapes ${e} and ${t}.`;throw Error(n)}else r[n-i-1]=a}return r}function al(e,t){let n=M(e,`a`,`equal`,`string_or_numeric`),r=M(t,`b`,`equal`,`string_or_numeric`);[n,r]=A(n,r),W(n.shape,r.shape);let i={a:n,b:r};return j.runKernel(Ot,i)}var ol=N({equal_:al});function sl(e,t,n){let r=M(t,`a`,`where`),i=M(n,`b`,`where`),a=M(e,`condition`,`where`,`bool`),o=W(W(a.shape,r.shape),i.shape),s={condition:dc(a,o),t:dc(r,o),e:dc(i,o)};return j.runKernel(Hn,s)}var cl=N({where_:sl});function ll(e){let t={x:M(e,`x`,`zerosLike`)};return j.runKernel(yr,t)}var ul=N({zerosLike_:ll});function dl(e,t){let n=M(e,`a`,`div`),r=M(t,`b`,`div`);[n,r]=A(n,r);let i=z(n,r),a=ul(i);return cl(ol(r,a),a,i)}var fl=N({divNoNan_:dl});function pl(e,t){let n=M(e,`t1`,`dot`),r=M(t,`t2`,`dot`);y((n.rank===1||n.rank===2)&&(r.rank===1||r.rank===2),()=>`Error in dot: inputs must all be rank 1 or 2, but got ranks ${n.rank} and ${r.rank}.`);let i=n.rank===1?n.size:n.shape[1],a=r.rank===1?r.size:r.shape[0];if(y(i===a,()=>`Error in dot: inner dimensions of inputs must match, but got ${i} and ${a}.`),n.rank===1&&r.rank===1)return V(H(V(n,[1,-1]),V(r,[-1,1])),[]);if(n.rank===1&&r.rank===2){let e=H(V(n,[1,-1]),V(r,[r.shape[0],r.shape[1]]));return V(e,[e.size])}else if(n.rank===2&&r.rank===1){let e=H(n,V(r,[-1,1]));return V(e,[e.size])}else return H(n,V(r,[r.shape[0],r.shape[1]]))}var ml=N({dot_:pl});function hl(e,...t){let n=t.map((e,t)=>M(e,`tensors${t}`,`einsum`)),r={equation:e};return j.runKernel(Et,n,r)}var gl=N({einsum_:hl});function _l(e){let t={x:M(e,`x`,`elu`,`float32`)};return j.runKernel(`Elu`,t)}var vl=N({elu_:_l});function yl(e,t){let n=M(e,`x`,`ensureShape`,`string_or_numeric`);if(!C(n.shape,t))throw Error(`EnsureShape: Shape of tensor ${n.shape} is not compatible with expected shape ${t}`);return e}var bl=N({ensureShape_:yl});function xl(e){let t=M(e,`x`,`erf`);y(t.dtype===`int32`||t.dtype===`float32`,()=>"Input dtype must be `int32` or `float32`."),t.dtype===`int32`&&(t=L(t,`float32`));let n={x:t};return j.runKernel(`Erf`,n)}var Sl=N({erf_:xl});function Cl(e,t){for(let n=0;n<e.length;++n)if(e[e.length-n-1]!==t-1-n)return!1;return!0}function wl(e,t,n){let r=e.length+t.length,i=[],a=0,o=0;for(let s=0;s<r;s++)n.indexOf(s)===-1?i.push(e[a++]):i.push(t[o++]);return i}function Tl(e,t){let n=[],r=e.length;for(let i=0;i<r;i++)t.indexOf(i)===-1&&n.push(e[i]);return[n,t.map(t=>e[t])]}function El(e,t){return wl(e,t.map(e=>1),t)}function Dl(e,t,n){y(Cl(t,n),()=>`${e} supports only inner-most axes for now. Got axes ${t} and rank-${n} input.`)}function Ol(e,t){if(Cl(e,t))return null;let n=[];for(let r=0;r<t;++r)e.indexOf(r)===-1&&n.push(r);return e.forEach(e=>n.push(e)),n}function kl(e){return e.map((e,t)=>[t,e]).sort((e,t)=>e[1]-t[1]).map(e=>e[0])}function Al(e,t){let n=[];for(let r=t-e;r<t;++r)n.push(r);return n}function jl(e,t=null,n=!1){let r={x:M(e,`x`,`max`)},i={reductionIndices:t,keepDims:n};return j.runKernel(`Max`,r,i)}var Ml=N({max_:jl});function Nl(e,t=null,n=!1){let r={x:M(e,`x`,`min`)},i={axis:t,keepDims:n};return j.runKernel(`Min`,r,i)}var Pl=N({min_:Nl});function Fl(e,t){let n=M(e,`base`,`pow`),r=M(t,`exp`,`pow`);[n,r]=A(n,r);let i={a:n,b:r};return j.runKernel(`Pow`,i)}var Il=N({pow_:Fl});function G(e,t){if((ai(e)&&t!==`string`||Array.isArray(e))&&t!==`complex64`)throw Error(`Error creating a new Scalar: value must be a primitive (number|boolean|string)`);if(t===`string`&&ai(e)&&!(e instanceof Uint8Array))throw Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return sa(e,[],[],t)}function Ll(e){let t={x:M(e,`x`,`sqrt`,`float32`)};return j.runKernel(Yn,t)}var Rl=N({sqrt_:Ll});function zl(e){let t=M(e,`x`,`square`);return j.runKernel(`Square`,{x:t},{})}var Bl=N({square_:zl});function Vl(e,t=null,n=!1){let r=M(e,`x`,`sum`);r.dtype===`bool`&&(r=L(r,`int32`));let i={x:r},a={axis:t,keepDims:n};return j.runKernel(`Sum`,i,a)}var K=N({sum_:Vl});function Hl(e,t=`euclidean`,n=null,r=!1){e=M(e,`x`,`norm`);let i=Ul(e,t,n),a=i.shape;if(r){let t=D(n,e.shape);a=El(i.shape,t)}return V(i,a)}function Ul(e,t,n=null){if(e.rank===0)return Ho(e);if(e.rank!==1&&n===null)return Ul(V(e,[-1]),t,n);if(e.rank===1||typeof n==`number`||Array.isArray(n)&&n.length===1){if(t===1)return K(Ho(e),n);if(t===1/0)return Ml(Ho(e),n);if(t===-1/0)return Pl(Ho(e),n);if(t===`euclidean`||t===2)return Rl(K(Il(Ho(e),G(2,`int32`)),n));throw Error(`Error in norm: invalid ord value: ${t}`)}if(Array.isArray(n)&&n.length===2){if(t===1)return Ml(K(Ho(e),n[0]),n[1]-1);if(t===1/0)return Ml(K(Ho(e),n[1]),n[0]);if(t===-1/0)return Pl(K(Ho(e),n[1]),n[0]);if(t===`fro`||t===`euclidean`)return Rl(K(Bl(e),n));throw Error(`Error in norm: invalid ord value: ${t}`)}throw Error(`Error in norm: invalid axis: ${n}`)}var Wl=N({norm_:Hl});function Gl(e,t=null,n=!1){return Wl(e,`euclidean`,t,n)}var Kl=N({euclideanNorm_:Gl});function ql(e){let t={x:M(e,`x`,`exp`)};return j.runKernel(`Exp`,t)}var Jl=N({exp_:ql});function Yl(e,t=0){let n=M(e,`x`,`expandDims`,`string_or_numeric`);y(t<=n.rank,()=>`Axis must be <= rank of the tensor`);let r={input:n},i={dim:t};return j.runKernel(kt,r,i)}var Xl=N({expandDims_:Yl});function Zl(e){let t={x:M(e,`x`,`expm1`)};return j.runKernel(At,t)}var Ql=N({expm1_:Zl});function $l(e,t){let n=M(e,`x`,`tile`,`string_or_numeric`);y(n.rank===t.length,()=>`Error in transpose: rank of input ${n.rank} must match length of reps ${t}.`);let r={x:n},i={reps:t};return j.runKernel(fr,r,i)}var eu=N({tile_:$l});function tu(e,t,n,r=`float32`){t??=e;let i=I([e,t],r),a=e<=t?e:t;for(let e=0;e<a;++e)i.set(1,e,e);let o=V(i.toTensor(),[e,t]);if(n==null)return o;if(n.length===1)return eu(Xl(o,0),[n[0],1,1]);if(n.length===2)return eu(Xl(Xl(o,0),0),[n[0],n[1],1,1]);if(n.length===3)return eu(Xl(Xl(Xl(o,0),0),0),[n[0],n[1],n[2],1,1]);throw Error(`eye() currently supports only 1D and 2D batchShapes, but received ${n.length}D.`)}var nu=N({eye_:tu});function ru(e){let t={x:M(e,`x`,`floor`,`float32`)};return j.runKernel(Nt,t)}var iu=N({floor_:ru});function au(e,t,n=0,r=0){let i={x:M(e,`x`,`gather`),indices:M(t,`indices`,`gather`,`int32`)},a={axis:n,batchDims:r};return j.runKernel(It,i,a)}var ou=N({gather_:au});function su(e,t){let n=M(e,`a`,`greater`,`string_or_numeric`),r=M(t,`b`,`greater`,`string_or_numeric`);[n,r]=A(n,r),W(n.shape,r.shape);let i={a:n,b:r};return j.runKernel(Rt,i)}var cu=N({greater_:su});function lu(e,t){let n=M(e,`a`,`greaterEqual`,`string_or_numeric`),r=M(t,`b`,`greaterEqual`,`string_or_numeric`);[n,r]=A(n,r),W(n.shape,r.shape);let i={a:n,b:r};return j.runKernel(zt,i)}var uu=N({greaterEqual_:lu});function du(e){let t={input:M(e,`input`,`imag`)};return j.runKernel(Ht,t)}var fu=N({imag_:du});function pu(e){let t={x:M(e,`x`,`isFinite`)};return j.runKernel(Ut,t)}var mu=N({isFinite_:pu});function hu(e){let t={x:M(e,`x`,`isInf`)};return j.runKernel(Wt,t)}var gu=N({isInf_:hu});function _u(e){let t={x:M(e,`x`,`isNaN`)};return j.runKernel(Gt,t)}var vu=N({isNaN_:_u});function yu(e,t=.2){let n={x:M(e,`x`,`leakyRelu`)},r={alpha:t};return j.runKernel(Kt,n,r)}var bu=N({leakyRelu_:yu});function xu(e,t){let n=M(e,`a`,`less`,`string_or_numeric`),r=M(t,`b`,`less`,`string_or_numeric`);[n,r]=A(n,r),W(n.shape,r.shape);let i={a:n,b:r};return j.runKernel(qt,i)}var Su=N({less_:xu});function Cu(e,t){let n=M(e,`a`,`lessEqual`,`string_or_numeric`),r=M(t,`b`,`lessEqual`,`string_or_numeric`);[n,r]=A(n,r),W(n.shape,r.shape);let i={a:n,b:r};return j.runKernel(Jt,i)}var wu=N({lessEqual_:Cu});function Tu(e,t,n){if(n<=0)throw Error(`The number of values should be positive.`);let r={start:e,stop:t,num:n};return j.runKernel(Yt,{},r)}function Eu(e,t=5,n=1,r=1,i=.5){let a=M(e,`x`,`localResponseNormalization`);y(a.rank===4||a.rank===3,()=>`Error in localResponseNormalization: x must be rank 3 or 4 but got
               rank ${a.rank}.`),y(ee(t),()=>`Error in localResponseNormalization: depthRadius must be an integer but got depthRadius ${t}.`);let o=a,s=!1;a.rank===3&&(s=!0,o=V(a,[1,a.shape[0],a.shape[1],a.shape[2]]));let c={x:o},l={depthRadius:t,bias:n,alpha:r,beta:i},u=j.runKernel(`LRN`,c,l);return s?V(u,[u.shape[1],u.shape[2],u.shape[3]]):u}var Du=N({localResponseNormalization_:Eu});function Ou(e){let t={x:M(e,`x`,`log`,`float32`)};return j.runKernel(`Log`,t)}var ku=N({log_:Ou});function Au(e){let t={x:M(e,`x`,`log1p`)};return j.runKernel(Xt,t)}var ju=N({log1p_:Au});function Mu(e,t){y(he(e),()=>`The f passed in variableGrads(f) must be a function`),y(t==null||Array.isArray(t)&&t.every(e=>e instanceof Oi),()=>`The varList passed in variableGrads(f, varList) must be an array of variables`);let n=t!=null;if(!n){t=[];for(let e in j.registeredVariables)t.push(j.registeredVariables[e])}let r=n?t.filter(e=>!e.trainable):null,i=t.length;t=t.filter(e=>e.trainable),y(t.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${i} variables is trainable.`);let{value:a,grads:o}=j.gradients(e,t,null,!0);y(o.some(e=>e!=null),()=>`Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize().`),y(a.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${a.rank} tensor`);let s={};return t.forEach((e,t)=>{o[t]!=null&&(s[e.name]=o[t])}),r?.forEach(e=>s[e.name]=null),{value:a,grads:s}}function Nu(e){return j.customGrad(e)}function Pu(e){let t={x:M(e,`x`,`neg`)};return j.runKernel(`Neg`,t)}var Fu=N({neg_:Pu});function Iu(e){let t={x:M(e,`x`,`softplus`)};return j.runKernel(Jn,t)}var Lu=N({softplus_:Iu});function Ru(e){let t=M(e,`x`,`logSigmoid`);return Nu(e=>({value:Fu(Lu(Fu(e))),gradFunc:t=>B(t,Vs(Fu(e)))}))(t)}var zu=N({logSigmoid_:Ru});function Bu(e,t){let n=M(e,`a`,`sub`),r=M(t,`b`,`sub`);[n,r]=A(n,r);let i={a:n,b:r};return j.runKernel(`Sub`,i)}var q=N({sub_:Bu});function Vu(e,t=-1){let n=M(e,`logits`,`logSoftmax`);if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and axis was ${t}`);return Nu((e,n)=>{let r=q(e,Ml(e,t,!0)),i=q(L(r,`float32`),ku(K(Jl(r),t,!0)));return n([i]),{value:i,gradFunc:(e,n)=>{let[r]=n,i=Jl(r);return q(e,B(K(e,t,!0),i))}}})(n)}var Hu=N({logSoftmax_:Vu});function Uu(e,t=null,n=!1){let r=M(e,`x`,`logSumExp`),i=D(t,r.shape),a=Ml(r,i,!0),o=ku(K(Jl(q(r,a)),i)),s=R(V(a,o.shape),o);return n?V(s,El(s.shape,i)):s}var Wu=N({logSumExp_:Uu});function Gu(e,t){let n=M(e,`a`,`logicalAnd`,`bool`),r=M(t,`b`,`logicalAnd`,`bool`);W(n.shape,r.shape);let i={a:n,b:r};return j.runKernel(Zt,i)}var Ku=N({logicalAnd_:Gu});function qu(e){let t={x:M(e,`x`,`logicalNot`,`bool`)};return j.runKernel(Qt,t)}var Ju=N({logicalNot_:qu});function Yu(e,t){let n=M(e,`a`,`logicalOr`,`bool`),r=M(t,`b`,`logicalOr`,`bool`);W(n.shape,r.shape);let i={a:n,b:r};return j.runKernel($t,i)}var Xu=N({logicalOr_:Yu});function Zu(e,t){let n=M(e,`a`,`logicalXor`,`bool`),r=M(t,`b`,`logicalXor`,`bool`);return W(n.shape,r.shape),Ku(Xu(e,t),Ju(Ku(e,t)))}var Qu=N({logicalXor_:Zu}),$u=2147483648;function ed(e,t,n=`left`){let r=M(e,`sortedSequence`,`searchSorted`),i=M(t,`values`,`searchSorted`),a=r.shape[r.shape.length-1],o=i.shape[i.shape.length-1],s=V(r,[-1,a]),c=V(i,[-1,o]);if(s.rank<2)throw Error(`Sorted input argument must be at least 2-dimensional`);if(s.shape[0]!==c.shape[0])throw Error(`Leading dimension of 'sortedSequence' and 'values' must match.`);if(S(c.shape)>=$u)throw Error(`values tensor size must less than ${$u}`);if(s.shape[1]>=$u)throw Error(`trailing dim_size must less than ${$u} for int32 output type, was ${s.shape[1]}`);let l={sortedSequence:s,values:c},u={side:n};return j.runKernel(Vn,l,u)}var td=N({searchSorted_:ed});function nd(e,t){return td(e,t,`left`)}function rd(e,t,n,r,i){let a=M(e,`x`,`maxPool`),o=a,s=!1;a.rank===3&&(s=!0,o=V(a,[1,a.shape[0],a.shape[1],a.shape[2]])),y(o.rank===4,()=>`Error in maxPool: input must be rank 4 but got rank ${o.rank}.`),y(Os(n,1),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${n} and dilations '1'`),js(`maxPool`,r,i);let c={x:o},l={filterSize:t,strides:n,pad:r,dimRoundingMode:i},u=j.runKernel(nn,c,l);return s?V(u,[u.shape[1],u.shape[2],u.shape[3]]):u}var id=N({maxPool_:rd});function ad(e,t=[1,1,1],n,r,i,a=`NDHWC`){let o=M(e,`x`,`maxPool3d`),s=o,c=!1;o.rank===4&&(c=!0,s=V(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),y(s.rank===5,()=>`Error in maxPool3d: x must be rank 5 but got rank ${s.rank}.`),y(a===`NDHWC`,()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),js(`maxPool3d`,r,i);let l={x:s},u={filterSize:t,strides:n,pad:r,dimRoundingMode:i,dataFormat:a},d=j.runKernel(an,l,u);return c?V(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}var od=N({maxPool3d_:ad});function sd(e,t,n,r,i=!1){let a={x:M(e,`x`,`maxPoolWithArgmax`)},o={filterSize:t,strides:n,pad:r,includeBatchInIndex:i},s=j.runKernel(sn,a,o);return{result:s[0],indexes:s[1]}}var cd=N({maxPoolWithArgmax_:sd});function ld(e,t){let n=M(e,`a`,`maximum`),r=M(t,`b`,`maximum`);[n,r]=A(n,r),n.dtype===`bool`&&(n=L(n,`int32`),r=L(r,`int32`)),W(n.shape,r.shape);let i={a:n,b:r};return j.runKernel(tn,i)}var ud=N({maximum_:ld});function dd(e,t=null,n=!1){let r={x:M(e,`x`,`mean`)},i={axis:t,keepDims:n};return j.runKernel(cn,r,i)}var fd=N({mean_:dd});function pd(e,t=`float32`){if(Se(e),t===`complex64`)return oa(pd(e,`float32`),pd(e,`float32`));let n=xe(S(e),t);return j.makeTensor(n,e,t)}function md(e,t=`float32`){if(Se(e),t===`complex64`)return oa(md(e,`float32`),pd(e,`float32`));let n=be(S(e),t);return j.makeTensor(n,e,t)}function hd(e,t,{indexing:n=`xy`}={}){if(n!==`xy`&&n!==`ij`)throw TypeError(`${n} is not a valid third argument to meshgrid`);if(e===void 0)return[];let r=M(e,`x`,`meshgrid`,e instanceof Ei?e.dtype:`float32`);if(t===void 0)return[r];let i=M(t,`y`,`meshgrid`,t instanceof Ei?t.dtype:`float32`),a=S(r.shape),o=S(i.shape);return n===`xy`?(r=V(r,[1,-1]),i=V(i,[-1,1]),[H(md([o,1],r.dtype),r),H(i,md([1,a],i.dtype))]):(r=V(r,[-1,1]),i=V(i,[1,-1]),[H(r,md([1,o],r.dtype)),H(md([a,1],i.dtype),i)])}function gd(e,t){let n=M(e,`a`,`minimum`),r=M(t,`b`,`minimum`);[n,r]=A(n,r),n.dtype===`bool`&&(n=L(n,`int32`),r=L(r,`int32`)),W(n.shape,r.shape);let i={a:n,b:r};return j.runKernel(ln,i)}var _d=N({minimum_:gd});function vd(e,t,n){y(n===`reflect`||n===`symmetric`,()=>`Invalid mode. Mode must be either reflect or symmetric. Got ${n}.`);let r=M(e,`x`,`mirrorPad`);if(r.rank===0)throw Error(`mirrorPad(scalar) is not defined. Pass non-scalar to mirrorPad`);y(t.length===r.rank,()=>`Padding doesn't match input. Must be ${r.rank}. Got ${t.length}.`);let i=+(n===`reflect`);for(let e=0;e<r.rank;e++)y(t[e].length===2,()=>`Invalid number of paddings. Must be length of 2 each.`),y(t[e][0]>=0&&t[e][0]<=r.shape[e]-i&&t[e][1]>=0&&t[e][1]<=r.shape[e]-i,()=>`Padding in dimension ${e} cannot be greater than or equal to ${r.shape[e]-i} or less than 0 for input of shape ${r.shape}`);let a={paddings:t,mode:n},o={x:r};return j.runKernel(un,o,a)}var yd=N({mirrorPad_:vd});function bd(e,t){let n=M(e,`a`,`mod`),r=M(t,`b`,`mod`);[n,r]=A(n,r);let i={a:n,b:r};return j.runKernel(`Mod`,i)}var xd=N({mod_:bd});function Sd(e,t=null,n=!1){e=M(e,`x`,`moments`);let r=D(t,e.shape),i=fd(e,r,n),a=i.shape;return n||(a=El(i.shape,r)),{mean:i,variance:fd(Bl(q(L(e,`float32`),V(i,a))),r,n)}}var Cd=N({moments_:Sd});function wd(e,t,n,r){let i=M(t,`data`,`multiRNNCell`),a=ra(n,`c`,`multiRNNCell`),o=ra(r,`h`,`multiRNNCell`),s=i,c=[];for(let t=0;t<e.length;t++){let n=e[t](s,a[t],o[t]);c.push(n[0]),c.push(n[1]),s=n[1]}let l=[],u=[];for(let e=0;e<c.length;e+=2)l.push(c[e]),u.push(c[e+1]);return[l,u]}var Td=N({multiRNNCell_:wd});function Ed(e,t,n,r=!1){let i=M(e,`logits`,`multinomial`),a=i.size,o=i.rank;if(a<2)throw Error(`Error in multinomial: you need at least 2 outcomes, but got ${a}.`);if(o>2)throw Error(`Rank of probabilities must be 1 or 2, but is ${o}`);n||=Math.random();let s={logits:o===1?V(i,[1,-1]):i},c={numSamples:t,seed:n,normalized:r},l=j.runKernel(dn,s,c);return o===1?V(l,[l.size]):l}var Dd=N({multinomial_:Ed});function Od(e,t){let n=M(e,`a`,`notEqual`,`string_or_numeric`),r=M(t,`b`,`notEqual`,`string_or_numeric`);[n,r]=A(n,r),W(n.shape,r.shape);let i={a:n,b:r};return j.runKernel(pn,i)}var kd=N({notEqual_:Od});function Ad(e,t,n=1,r=0,i=`int32`){if(t<2)throw Error(`Error in oneHot: depth must be >=2, but it is ${t}`);let a={indices:M(e,`indices`,`oneHot`,`int32`)},o={dtype:i,depth:t,onValue:n,offValue:r};return j.runKernel(vn,a,o)}var jd=N({oneHot_:Ad});function Md(e){let t={x:M(e,`x`,`onesLike`)};return j.runKernel(_n,t)}var Nd=N({onesLike_:Md});function Pd(e,t){let n=M(e,`v1`,`outerProduct`),r=M(t,`v2`,`outerProduct`);return y(n.rank===1&&r.rank===1,()=>`Error in outerProduct: inputs must be rank 1, but got ranks ${n.rank} and ${r.rank}.`),H(V(n,[-1,1]),V(r,[1,-1]))}var Fd=N({outerProduct_:Pd});function Id(e,t,n=0){let r=M(e,`x`,`pad`);if(r.rank===0)throw Error(`pad(scalar) is not defined. Pass non-scalar to pad`);let i={paddings:t,constantValue:n},a={x:r};return j.runKernel(bn,a,i)}var Ld=N({pad_:Id});function Rd(e,t,n=0){return y(t.length===2,()=>`Invalid number of paddings. Must be length of 2.`),Ld(e,[t],n)}var zd=N({pad1d_:Rd});function Bd(e,t,n=0){return y(t.length===2&&t[0].length===2&&t[1].length===2,()=>`Invalid number of paddings. Must be length of 2 each.`),Ld(e,t,n)}var Vd=N({pad2d_:Bd});function Hd(e,t,n=0){return y(t.length===3&&t[0].length===2&&t[1].length===2&&t[2].length===2,()=>`Invalid number of paddings. Must be length of 2 each.`),Ld(e,t,n)}var Ud=N({pad3d_:Hd});function Wd(e,t,n=0){return y(t.length===4&&t[0].length===2&&t[1].length===2&&t[2].length===2&&t[3].length===2,()=>`Invalid number of paddings. Must be length of 2 each.`),Ld(e,t,n)}var Gd=N({pad4d_:Wd});function Kd(e,t,n){let r=M(e,`x`,`spaceToBatchND`);y(r.rank>=1+t.length,()=>`input rank ${r.rank} should be > than [blockShape] ${t.length}`),y(n.length===t.length,()=>`paddings.shape[0] ${n.length} must be equal to [blockShape] ${t.length}`),y(r.shape.reduce((e,r,i)=>i>0&&i<=t.length?e&&(r+n[i-1][0]+n[i-1][1])%t[i-1]===0:e,!0),()=>`input spatial dimensions ${r.shape.slice(1)} with paddings ${n.toString()} must be divisible by blockShapes ${t.toString()}`);let i={x:r},a={blockShape:t,paddings:n};return j.runKernel(Xn,i,a)}var qd=N({spaceToBatchND_:Kd});function Jd(e,t,n,r,i,a,o){i??=[1,1],a??=1,r===0&&(r=`valid`);let s=M(e,`x`,`maxPool`),c=s,l=!1;s.rank===3&&(l=!0,c=V(s,[1,s.shape[0],s.shape[1],s.shape[2]])),y(Os(a,i),()=>`Error in pool: Either strides or dilations must be 1. Got strides ${a} and dilations '${i}'`);let u=ms(c.shape,t,a,i,r),d=[u.dilationHeight,u.dilationWidth],f;f=r===`same`?Xd([u.filterHeight,u.filterWidth],d):[[0,0],[0,0]];let p=d[0]===1&&d[1]===1,[m,h]=Yd([u.inHeight,u.inWidth],d,f),g=p?r:`valid`,_=p?c:qd(c,d,m),v=(n===`avg`?()=>Ps(_,t,a,g,o):()=>id(_,t,a,g,o))(),b=p?v:Js(v,d,h);return l?V(b,[b.shape[1],b.shape[2],b.shape[3]]):b}function Yd(e,t,n){let r=n.map(e=>e[0]),i=n.map(e=>e[1]),a=e.concat(r,i),o=t.map((e,t)=>(e-a[t]%e)%e),s=i.map((e,t)=>e+o[t]);return[t.map((e,t)=>[r[t],s[t]]),t.map((e,t)=>[0,o[t]])]}function Xd(e,t){let n=e.map((e,n)=>e+(e-1)*(t[n]-1)).map(e=>e-1),r=n.map(e=>Math.floor(e/2)),i=n.map((e,t)=>e-r[t]);return n.map((e,t)=>[r[t],i[t]])}var Zd=N({pool_:Jd});function Qd(e,t){let n={x:M(e,`x`,`prelu`),alpha:M(t,`alpha`,`prelu`)};return j.runKernel(xn,n)}var $d=N({prelu_:Qd});function ef(e,t=null,n=!1){let r=M(e,`x`,`prod`);r.dtype===`bool`&&(r=L(r,`int32`));let i={x:r},a={axis:t,keepDims:n};return j.runKernel(Sn,i,a)}var tf=N({prod_:ef});function nf(e,t,n,r){let i={paramsNestedSplits:e.map((e,t)=>M(e,`tensors${t}`,`raggedGather`,`int32`)),paramsDenseValues:M(t,`paramsDenseValues`,`raggedGather`),indices:M(n,`indices`,`raggedGather`,`int32`)},a={outputRaggedRank:r},o=j.runKernel(Cn,i,a);return{outputNestedSplits:o.slice(0,o.length-1),outputDenseValues:o[o.length-1]}}var rf=N({raggedGather_:nf});function af(e,t,n){let r=M(e,`starts`,`raggedRange`),i={starts:r,limits:M(t,`limits`,`raggedRange`,r.dtype),deltas:M(n,`deltas`,`raggedRange`,r.dtype)},a=j.runKernel(wn,i);return{rtNestedSplits:a[0],rtDenseValues:a[1]}}var of=N({raggedRange_:af});function sf(e,t,n,r,i){let a=M(e,`shape`,`raggedTensorToTensor`,`int32`),o=M(t,`values`,`raggedTensorToTensor`),s={shape:a,values:o,defaultValue:M(n,`defaultValue`,`raggedTensorToTensor`,o.dtype),rowPartitionTensors:r.map((e,t)=>M(e,`tensors${t}`,`raggedTensorToTensor`,`int32`))},c={rowPartitionTypes:i};return j.runKernel(Tn,s,c)}var cf=N({raggedTensorToTensor_:sf});function lf(e,t,n){Se(e);let r=S(e),i=null;if(n==null||n===`float32`)i=new Float32Array(r);else if(n===`int32`)i=new Int32Array(r);else if(n===`bool`)i=new Uint8Array(r);else throw Error(`Unknown data type ${n}`);for(let e=0;e<r;e++)i[e]=t();return j.makeTensor(i,e,n)}var uf=N({rand_:lf}),df=o(((e,t)=>{(function(e,t,n){function r(e){var t=this,n=o();t.next=function(){var e=2091639*t.s0+t.c*23283064365386963e-26;return t.s0=t.s1,t.s1=t.s2,t.s2=e-(t.c=e|0)},t.c=1,t.s0=n(` `),t.s1=n(` `),t.s2=n(` `),t.s0-=n(e),t.s0<0&&(t.s0+=1),t.s1-=n(e),t.s1<0&&(t.s1+=1),t.s2-=n(e),t.s2<0&&(t.s2+=1),n=null}function i(e,t){return t.c=e.c,t.s0=e.s0,t.s1=e.s1,t.s2=e.s2,t}function a(e,t){var n=new r(e),a=t&&t.state,o=n.next;return o.int32=function(){return n.next()*4294967296|0},o.double=function(){return o()+(o()*2097152|0)*11102230246251565e-32},o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}function o(){var e=4022871197;return function(t){t=String(t);for(var n=0;n<t.length;n++){e+=t.charCodeAt(n);var r=.02519603282416938*e;e=r>>>0,r-=e,r*=e,e=r>>>0,r-=e,e+=r*4294967296}return(e>>>0)*23283064365386963e-26}}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.alea=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),ff=o(((e,t)=>{(function(e,t,n){function r(e){var t=this,n=``;t.x=0,t.y=0,t.z=0,t.w=0,t.next=function(){var e=t.x^t.x<<11;return t.x=t.y,t.y=t.z,t.z=t.w,t.w^=t.w>>>19^e^e>>>8},e===(e|0)?t.x=e:n+=e;for(var r=0;r<n.length+64;r++)t.x^=n.charCodeAt(r)|0,t.next()}function i(e,t){return t.x=e.x,t.y=e.y,t.z=e.z,t.w=e.w,t}function a(e,t){var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xor128=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),pf=o(((e,t)=>{(function(e,t,n){function r(e){var t=this,n=``;t.next=function(){var e=t.x^t.x>>>2;return t.x=t.y,t.y=t.z,t.z=t.w,t.w=t.v,(t.d=t.d+362437|0)+(t.v=t.v^t.v<<4^(e^e<<1))|0},t.x=0,t.y=0,t.z=0,t.w=0,t.v=0,e===(e|0)?t.x=e:n+=e;for(var r=0;r<n.length+64;r++)t.x^=n.charCodeAt(r)|0,r==n.length&&(t.d=t.x<<10^t.x>>>4),t.next()}function i(e,t){return t.x=e.x,t.y=e.y,t.z=e.z,t.w=e.w,t.v=e.v,t.d=e.d,t}function a(e,t){var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xorwow=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),mf=o(((e,t)=>{(function(e,t,n){function r(e){var t=this;t.next=function(){var e=t.x,n=t.i,r=e[n],i;return r^=r>>>7,i=r^r<<24,r=e[n+1&7],i^=r^r>>>10,r=e[n+3&7],i^=r^r>>>3,r=e[n+4&7],i^=r^r<<7,r=e[n+7&7],r^=r<<13,i^=r^r<<9,e[n]=i,t.i=n+1&7,i};function n(e,t){var n,r=[];if(t===(t|0))r[0]=t;else for(t=``+t,n=0;n<t.length;++n)r[n&7]=r[n&7]<<15^t.charCodeAt(n)+r[n+1&7]<<13;for(;r.length<8;)r.push(0);for(n=0;n<8&&r[n]===0;++n);for(n==8?r[7]=-1:r[n],e.x=r,e.i=0,n=256;n>0;--n)e.next()}n(t,e)}function i(e,t){return t.x=e.x.slice(),t.i=e.i,t}function a(e,t){e??=+new Date;var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(a.x&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xorshift7=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),hf=o(((e,t)=>{(function(e,t,n){function r(e){var t=this;t.next=function(){var e=t.w,n=t.X,r=t.i,i,a;return t.w=e=e+1640531527|0,a=n[r+34&127],i=n[r=r+1&127],a^=a<<13,i^=i<<17,a^=a>>>15,i^=i>>>12,a=n[r]=a^i,t.i=r,a+(e^e>>>16)|0};function n(e,t){var n,r,i,a,o,s=[],c=128;for(t===(t|0)?(r=t,t=null):(t+=`\0`,r=0,c=Math.max(c,t.length)),i=0,a=-32;a<c;++a)t&&(r^=t.charCodeAt((a+32)%t.length)),a===0&&(o=r),r^=r<<10,r^=r>>>15,r^=r<<4,r^=r>>>13,a>=0&&(o=o+1640531527|0,n=s[a&127]^=r+o,i=n==0?i+1:0);for(i>=128&&(s[(t&&t.length||0)&127]=-1),i=127,a=512;a>0;--a)r=s[i+34&127],n=s[i=i+1&127],r^=r<<13,n^=n<<17,r^=r>>>15,n^=n>>>12,s[i]=r^n;e.w=o,e.X=s,e.i=i}n(t,e)}function i(e,t){return t.i=e.i,t.w=e.w,t.X=e.X.slice(),t}function a(e,t){e??=+new Date;var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(a.X&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xor4096=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),gf=o(((e,t)=>{(function(e,t,n){function r(e){var t=this,n=``;t.next=function(){var e=t.b,n=t.c,r=t.d,i=t.a;return e=e<<25^e>>>7^n,n=n-r|0,r=r<<24^r>>>8^i,i=i-e|0,t.b=e=e<<20^e>>>12^n,t.c=n=n-r|0,t.d=r<<16^n>>>16^i,t.a=i-e|0},t.a=0,t.b=0,t.c=-1640531527,t.d=1367130551,e===Math.floor(e)?(t.a=e/4294967296|0,t.b=e|0):n+=e;for(var r=0;r<n.length+20;r++)t.b^=n.charCodeAt(r)|0,t.next()}function i(e,t){return t.a=e.a,t.b=e.b,t.c=e.c,t.d=e.d,t}function a(e,t){var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.tychei=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),_f=o(((e,t)=>{(function(e,n,r){var i=256,a=6,o=52,s=`random`,c=r.pow(i,a),l=r.pow(2,o),u=l*2,d=i-1,f;function p(e,t,o){var d=[];t=t==1?{entropy:!0}:t||{};var f=_(g(t.entropy?[e,y(n)]:e??v(),3),d),p=new m(d),b=function(){for(var e=p.g(a),t=c,n=0;e<l;)e=(e+n)*i,t*=i,n=p.g(1);for(;e>=u;)e/=2,t/=2,n>>>=1;return(e+n)/t};return b.int32=function(){return p.g(4)|0},b.quick=function(){return p.g(4)/4294967296},b.double=b,_(y(p.S),n),(t.pass||o||function(e,t,n,i){return i&&(i.S&&h(i,p),e.state=function(){return h(p,{})}),n?(r[s]=e,t):e})(b,f,`global`in t?t.global:this==r,t.state)}function m(e){var t,n=e.length,r=this,a=0,o=r.i=r.j=0,s=r.S=[];for(n||(e=[n++]);a<i;)s[a]=a++;for(a=0;a<i;a++)s[a]=s[o=d&o+e[a%n]+(t=s[a])],s[o]=t;(r.g=function(e){for(var t,n=0,a=r.i,o=r.j,s=r.S;e--;)t=s[a=d&a+1],n=n*i+s[d&(s[a]=s[o=d&o+t])+(s[o]=t)];return r.i=a,r.j=o,n})(i)}function h(e,t){return t.i=e.i,t.j=e.j,t.S=e.S.slice(),t}function g(e,t){var n=[],r=typeof e,i;if(t&&r==`object`)for(i in e)try{n.push(g(e[i],t-1))}catch{}return n.length?n:r==`string`?e:e+`\0`}function _(e,t){for(var n=e+``,r,i=0;i<n.length;)t[d&i]=d&(r^=t[d&i]*19)+n.charCodeAt(i++);return y(t)}function v(){try{var t;return f&&(t=f.randomBytes)?t=t(i):(t=new Uint8Array(i),(e.crypto||e.msCrypto).getRandomValues(t)),y(t)}catch{var r=e.navigator,a=r&&r.plugins;return[+new Date,e,a,e.screen,y(n)]}}function y(e){return String.fromCharCode.apply(0,e)}if(_(r.random(),n),typeof t==`object`&&t.exports){t.exports=p;try{f=Oo()}catch{}}else typeof define==`function`&&define.amd?define(function(){return p}):r[`seed`+s]=p})(typeof self<`u`?self:e,[],Math)})),vf=l(o(((e,t)=>{var n=df(),r=ff(),i=pf(),a=mf(),o=hf(),s=gf(),c=_f();c.alea=n,c.xor128=r,c.xorwow=i,c.xorshift7=a,c.xor4096=o,c.tychei=s,t.exports=c}))()),yf=class{constructor(e,t,n,r,i){this.mean=e,this.stdDev=t,this.dtype=n,this.nextVal=NaN,this.truncated=r,this.truncated&&(this.upper=this.mean+this.stdDev*2,this.lower=this.mean-this.stdDev*2);let a=i||Math.random();this.random=vf.alea(a.toString())}nextValue(){if(!isNaN(this.nextVal)){let e=this.nextVal;return this.nextVal=NaN,e}let e,t,n=!1;for(;!n;){let r,i,a;do r=2*this.random()-1,i=2*this.random()-1,a=r*r+i*i;while(a>=1||a===0);let o=Math.sqrt(-2*Math.log(a)/a);e=this.mean+this.stdDev*r*o,t=this.mean+this.stdDev*i*o,(!this.truncated||this.isValidTruncated(e))&&(n=!0)}return(!this.truncated||this.isValidTruncated(t))&&(this.nextVal=this.convertValue(t)),this.convertValue(e)}convertValue(e){return this.dtype==null||this.dtype===`float32`?e:Math.round(e)}isValidTruncated(e){return e<=this.upper&&e>=this.lower}},bf=class{constructor(e,t,n,r){this.alpha=e,this.beta=1/t,this.dtype=n;let i=r||Math.random();this.randu=vf.alea(i.toString()),this.randn=new yf(0,1,n,!1,this.randu()),e<1?this.d=e+2/3:this.d=e-1/3,this.c=1/Math.sqrt(9*this.d)}nextValue(){let e,t,n,r,i,a;for(;;){do r=this.randn.nextValue(),a=1+this.c*r;while(a<=0);if(a*=a*a,e=r*r,t=1-.331*e*e,n=.5*e+this.d*(1-a+Math.log(a)),i=this.randu(),i<t||Math.log(i)<n)break}return a=1/this.beta*this.d*a,this.alpha<1&&(a*=this.randu()**(1/this.alpha)),this.convertValue(a)}convertValue(e){return this.dtype===`float32`?e:Math.round(e)}},xf=class{constructor(e=0,t=1,n,r){if(this.canReturnFloat=()=>this.dtype==null||this.dtype===`float32`,this.min=e,this.range=t-e,this.dtype=n,r??=Math.random(),typeof r==`number`&&(r=r.toString()),!this.canReturnFloat()&&this.range<=1)throw Error(`The difference between ${e} - ${t} <= 1 and dtype is not float`);this.random=vf.alea(r)}convertValue(e){return this.canReturnFloat()?e:Math.round(e)}nextValue(){return this.convertValue(this.min+this.range*this.random())}};function Sf(e,t,n=1,r=`float32`,i){if(Se(e),n??=1,r??=`float32`,r!==`float32`&&r!==`int32`)throw Error(`Unsupported data type ${r}`);let a=new bf(t,n,r,i),o=I(e,r);for(let e=0;e<o.values.length;e++)o.values[e]=a.nextValue();return o.toTensor()}var Cf=N({randomGamma_:Sf});function wf(e,t=0,n=1,r,i){if(Se(e),r!=null&&r===`bool`)throw Error(`Unsupported data type ${r}`);let a=new yf(t,n,r,!1,i),o=I(e,r);for(let e=0;e<o.values.length;e++)o.values[e]=a.nextValue();return o.toTensor()}var Tf=N({randomNormal_:wf});function Ef(e,t,n){if(t!=null&&t===`bool`)throw Error(`Unsupported data type ${t}`);return Tf(e,0,1,t,n)}var Df=N({randomStandardNormal_:Ef});function Of(e,t=0,n=1,r=`float32`,i){Se(e);let a=I(e,r),o=new xf(t,n,null,i);for(let e=0;e<a.values.length;e++)a.values[e]=o.nextValue();return a.toTensor()}var kf=N({randomUniform_:Of});function Af(e,t,n,r){return kf(e,t,n,`int32`,r)}var jf=N({randomUniformInt_:Af});function Mf(e,t,n=1,r=`float32`){if(n===0)throw Error(`Cannot have a step of zero`);let i={start:e,stop:t,step:n,dtype:r};return j.runKernel(En,{},i)}function Nf(e){let t={input:M(e,`input`,`real`)};return j.runKernel(Dn,t)}var Pf=N({real_:Nf});function Ff(e){let t={x:M(e,`x`,`reciprocal`)};return j.runKernel(On,t)}var If=N({reciprocal_:Ff});function Lf(e){let t={x:M(e,`x`,`relu`)};return j.runKernel(kn,t)}var Rf=N({relu_:Lf});function zf(e){let t={x:M(e,`x`,`relu6`)};return j.runKernel(Fn,t)}var Bf=N({relu6_:zf});function Vf(e,t){let n={x:M(e,`x`,`reverse`)},r={dims:t};return j.runKernel(In,n,r)}var Hf=N({reverse_:Vf});function Uf(e){let t=M(e,`x`,`reverse`);return y(t.rank===1,()=>`Error in reverse1D: x must be rank 1 but got rank ${t.rank}.`),Hf(t,0)}var Wf=N({reverse1d_:Uf});function Gf(e,t){let n=M(e,`x`,`reverse`);return y(n.rank===2,()=>`Error in reverse2D: x must be rank 2 but got rank ${n.rank}.`),Hf(n,t)}var Kf=N({reverse2d_:Gf});function qf(e,t){let n=M(e,`x`,`reverse`);return y(n.rank===3,()=>`Error in reverse3D: x must be rank 3 but got rank ${n.rank}.`),Hf(n,t)}var Jf=N({reverse3d_:qf});function Yf(e,t){let n=M(e,`x`,`reverse`);return y(n.rank===4,()=>`Error in reverse4D: x must be rank 4 but got rank ${n.rank}.`),Hf(n,t)}var Xf=N({reverse4d_:Yf});function Zf(e){let t={x:M(e,`x`,`round`)};return j.runKernel(Ln,t)}var Qf=N({round_:Zf});function $f(e){let t={x:M(e,`x`,`rsqrt`,`float32`)};return j.runKernel(Rn,t)}var ep=N({rsqrt_:$f});function tp(e){let t={x:M(e,`x`,`selu`)};return j.runKernel(Un,t)}var np=N({selu_:tp});function rp(e,t,n,r,i,a=[1,1],o=`NHWC`){let s=M(e,`x`,`separableConv2d`),c=M(t,`depthwiseFilter`,`separableConv2d`),l=M(n,`pointwiseFilter`,`separableConv2d`),u=s,d=!1;if(s.rank===3&&(d=!0,u=V(s,[1,s.shape[0],s.shape[1],s.shape[2]])),o===`NCHW`)throw Error(`separableConv2d currently does not support dataFormat NCHW; only NHWC is supported`);y(u.rank===4,()=>`Error in separableConv2d: input must be rank 4, but got rank ${u.rank}.`),y(c.rank===4,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${c.rank}.`),y(l.rank===4,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${c.rank}.`),y(l.shape[0]===1,()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${l.shape[0]}.`),y(l.shape[1]===1,()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${l.shape[1]}.`);let f=c.shape[2],p=c.shape[3];y(l.shape[2]===f*p,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${f*p}, but got ${l.shape[2]}.`);let m=Ec(Qc(u,c,r,i,o,a),l,1,`valid`,o);return d?V(m,[m.shape[1],m.shape[2],m.shape[3]]):m}var ip=N({separableConv2d_:rp});async function ap(e,t){let n=M(e,`x`,`setdiff1d`),r=M(t,`y`,`setdiff1d`);y(n.dtype===r.dtype,()=>`x and y should have the same dtype, but got x (${n.dtype}) and y (${r.dtype}).`),y(n.rank===1,()=>`x should be 1D tensor, but got x (${n.shape}).`),y(r.rank===1,()=>`y should be 1D tensor, but got y (${r.shape}).`);let i=await n.data(),a=await r.data(),o=new Set(a),s=0;for(let e=0;e<i.length;e++)o.has(i[e])||s++;let c=new xi([s],n.dtype),l=new xi([s],`int32`);for(let e=0,t=0;e<i.length;e++)o.has(i[e])||(c.values[t]=i[e],l.values[t]=e,t++);return[c.toTensor(),l.toTensor()]}var op=ap;function sp(e){let t={x:M(e,`x`,`sign`)};return j.runKernel(Kn,t)}var cp=N({sign_:sp});function lp(e){let t={x:M(e,`x`,`sin`,`float32`)};return j.runKernel(`Sin`,t)}var up=N({sin_:lp});function dp(e){let t={x:M(e,`x`,`sinh`)};return j.runKernel(Gn,t)}var fp=N({sinh_:dp});function pp(e,t,n){let r=M(e,`x`,`slice1d`);return y(r.rank===1,()=>`slice1d expects a rank-1 tensor, but got a rank-${r.rank} tensor`),U(r,[t],[n])}var mp=N({slice1d_:pp});function hp(e,t,n){let r=M(e,`x`,`slice2d`);return y(r.rank===2,()=>`slice2d expects a rank-2 tensor, but got a rank-${r.rank} tensor`),U(r,t,n)}var gp=N({slice2d_:hp});function _p(e,t,n){let r=M(e,`x`,`slice3d`);return y(r.rank===3,()=>`slice3d expects a rank-3 tensor, but got a rank-${r.rank} tensor`),U(r,t,n)}var vp=N({slice3d_:_p});function yp(e,t,n){let r=M(e,`x`,`slice4d`);return y(r.rank===4,()=>`slice4d expects a rank-4 tensor, but got a rank-${r.rank} tensor`),U(r,t,n)}var bp=N({slice4d_:yp});function xp(e,t=-1){let n=M(e,`logits`,`softmax`,`float32`);if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and dim was ${t}`);let r={logits:n},i={dim:t};return j.runKernel(Qn,r,i)}var Sp=N({softmax_:xp});function Cp(e){y(e.dtype===`complex64`,()=>`The dtype for tf.spectral.fft() must be complex64 but got ${e.dtype}.`);let t={input:e};return j.runKernel(`FFT`,t)}var wp=N({fft_:Cp});function Tp(e){y(e.dtype===`complex64`,()=>`The dtype for tf.spectral.ifft() must be complex64 but got ${e.dtype}.`);let t={input:e};return j.runKernel(Vt,t)}var Ep=N({ifft_:Tp});function Dp(e){let t=e.shape[e.shape.length-1],n=e.size/t,r;if(t<=2)r=Ep(V(e,[n,t]));else{let i=[n,2*(t-1)],a=V(Pf(e),[n,t]),o=V(fu(e),[n,t]),s=Hf(U(a,[0,1],[n,t-2]),1),c=B(Hf(U(o,[0,1],[n,t-2]),1),G(-1));r=Ep(V(oa(Rs([a,s],1),Rs([o,c],1)),[i[0],i[1]]))}if(r=Pf(r),e.rank===3&&e.shape[0]!==0){let t=r,n=e.shape[0];r=V(r,[n,r.shape[0]/n,r.shape[1]]),t.dispose()}return r}var Op=N({irfft_:Dp});function kp(e,t,n=0){let r={x:M(e,`x`,`split`)},i={numOrSizeSplits:t,axis:n};return j.runKernel(Zn,r,i)}var Ap=N({split_:kp});function jp(e,t){y(e.dtype===`float32`,()=>`The dtype for rfft() must be real value but got ${e.dtype}`);let n=e.shape[e.shape.length-1],r=e.size/n,i;if(t!=null&&t<n){let r=e.shape.map(e=>0),a=e.shape.map(e=>e);a[e.shape.length-1]=t,i=U(e,r,a),n=t}else if(t!=null&&t>n){let r=e.shape.map(e=>e);r[e.shape.length-1]=t-n,i=Rs([e,pd(r)],e.shape.length-1),n=t}else i=e;let a=ul(i),o=wp(V(oa(i,a),[r,n])),s=Math.floor(n/2)+1,c=Pf(o),l=fu(o),u=Ap(c,[s,n-s],c.shape.length-1),d=Ap(l,[s,n-s],l.shape.length-1),f=i.shape.slice();return f[i.shape.length-1]=s,V(oa(u[0],d[0]),f)}var Mp=N({rfft_:jp});function Np(e,t){let n=M(e,`a`,`squaredDifference`),r=M(t,`b`,`squaredDifference`);[n,r]=A(n,r),W(n.shape,r.shape);let i={a:n,b:r};return j.runKernel(ir,i,{})}var Pp=N({squaredDifference_:Np});function Fp(e,t){let n=M(e,`x`,`squeeze`,`string_or_numeric`);return V(n,re(n.shape,t).newShape)}var Ip=N({squeeze_:Fp});function Lp(e,t=0){let n=ra(e,`tensors`,`stack`,`string_or_numeric`);y(n.length>=1,()=>`Pass at least one tensor to tf.stack`),n.length>0&&y(t<=n[0].rank,()=>`Axis must be <= rank of the tensor`);let r=n,i={axis:t};return j.runKernel(yn,r,i)}var Rp=N({stack_:Lp});function zp(e,t=0){let n={x:M(e,`x`,`step`)},r={alpha:t};return j.runKernel(br,n,r)}var Bp=N({step_:zp});function Vp(e,t,n,r,i=0,a=0,o=0,s=0,c=0){let l={x:M(e,`x`,`stridedSlice`,`string_or_numeric`)},u={begin:t,end:n,strides:r,beginMask:i,endMask:a,ellipsisMask:o,newAxisMask:s,shrinkAxisMask:c};return j.runKernel(sr,l,u)}var Hp=N({stridedSlice_:Vp});function Up(e){let t={x:M(e,`x`,`tan`,`float32`)};return j.runKernel(`Tan`,t)}var Wp=N({tan_:Up});function Gp(e,t){x(e);let n=ea(e,t);if(n.length!==1)throw Error(`tensor1d() requires values to be a flat/TypedArray`);return sa(e,null,n,t)}function Kp(e,t,n){if(x(e),t!=null&&t.length!==2)throw Error(`tensor2d() requires shape to have two numbers`);let r=ea(e,n);if(r.length!==2&&r.length!==1)throw Error(`tensor2d() requires values to be number[][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return sa(e,t,r,n)}function qp(e,t,n){if(x(e),t!=null&&t.length!==3)throw Error(`tensor3d() requires shape to have three numbers`);let r=ea(e,n);if(r.length!==3&&r.length!==1)throw Error(`tensor3d() requires values to be number[][][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor3d() requires shape to be provided when `values` are a flat array");return sa(e,t,r,n)}function Jp(e,t,n){if(x(e),t!=null&&t.length!==4)throw Error(`tensor4d() requires shape to have four numbers`);let r=ea(e,n);if(r.length!==4&&r.length!==1)throw Error(`tensor4d() requires values to be number[][][][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor4d() requires shape to be provided when `values` are a flat array");return sa(e,t,r,n)}function Yp(e,t,n){if(x(e),t!=null&&t.length!==5)throw Error(`tensor5d() requires shape to have five numbers`);let r=ea(e,n);if(r.length!==5&&r.length!==1)throw Error(`tensor5d() requires values to be number[][][][][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor5d() requires shape to be provided when `values` are a flat array");return sa(e,t,r,n)}function Xp(e,t,n){if(x(e),t!=null&&t.length!==6)throw Error(`tensor6d() requires shape to have six numbers`);let r=ea(e,n);if(r.length!==6&&r.length!==1)throw Error(`tensor6d() requires values to be number[][][][][][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor6d() requires shape to be provided when `values` are a flat array");return t||=r,sa(e,t,r,n)}function Zp(e,t,n){let r=t.rank>1?t.shape[t.rank-1]:1,i=t.rank>1?t.rank-1:1,a=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${n.shape}, indices.shape: ${t.shape}, shape: ${e}, sliceDim: ${r}, and batchDim: ${i}.`;if(n.rank<i)throw Error(a+` update.rank < ${i}. `);if(e.length<r+(n.rank-i))throw Error(a+` Output shape length < ${r+(n.rank-i)}`);if(n.rank!==i+e.length-r)throw Error(a+` update.rank != ${i+e.length-r}`);for(let e=0;e<i;++e)if(n.shape[e]!==t.shape[e])throw Error(a+` updates.shape[${e}] (${n.shape[e]}) != indices.shape[${e}] (${t.shape[e]}).`);for(let t=0;t<n.rank-i;++t)if(n.shape[t+i]!==e[t+r])throw Error(a+` updates.shape[${t+i}] (${n.shape[t+i]}) != shape[${t+i}] (${e[t+i]})`)}function Qp(e,t,n){if(t.rank<1)throw Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${t.rank}.`);if(e.rank<1)throw Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${e.rank}.`);if(t.dtype!==`int32`)throw Error(`The dtype of 'indices' should be int32, but got dtype: ${t.dtype}`);if(n.length<1)throw Error(`Output rank must be greater or equal to 1, but got shape: ${n}`);if(n.length===0){if(t.size===0)throw Error(`Indices specified for empty output. indices shape: ${t.shape}`);if(e.size===0)throw Error(`Updates specified for empty output. updates shape: ${e.shape}`)}Zp(n,t,e)}function $p(e,t,n){let r=t.shape.length,i=r>1?t.shape[r-1]:1,a=n.length,o=1;for(let e=i;e<a;++e)o*=n[e];let s=i<1?1:i,c=S(t.shape)/s,l=[..._e(n.slice(0,i)),1],u=S(n);return{sliceRank:i,numUpdates:c,sliceSize:o,strides:l,outputSize:u}}function em(e,t,n){let r=M(e,`tensor`,`tensorScatterupdate`),i=M(t,`indices`,`tensorScatterupdate`,`int32`),a=M(n,`updates`,`tensorScatterupdate`);if(Qp(a,i,r.shape),r.dtype!==a.dtype)throw Error(`tensor and updates must have the same dtype, instead they are ${r.dtype} and ${a.dtype}.`);let o={tensor:r,indices:i,updates:a};return j.runKernel(Bn,o,{})}var tm=N({tensorScatterUpdate_:em});function nm(e,t=1,n=!0){let r=M(e,`x`,`topk`);if(r.rank===0)throw Error(`topk() expects the input to be of rank 1 or higher`);let i=r.shape[r.shape.length-1];if(t<0)throw Error(`'k' passed to topk() must be >= 0 but got ${t}`);if(t>i)throw Error(`'k' passed to topk() must be <= the last dimension (${i}) but got ${t}`);let a={x:r},o={k:t,sorted:n},[s,c]=j.runKernel(pr,a,o);return{values:s,indices:c}}var rm=N({topk_:nm});function im(e,t=0,n=1,r,i){if(Se(e),r!=null&&r===`bool`)throw Error(`Unsupported data type $ { dtype }`);let a=new yf(t,n,r,!0,i),o=I(e,r);for(let e=0;e<o.values.length;e++)o.values[e]=a.nextValue();return o.toTensor()}var am=N({truncatedNormal_:im});function om(e,t=0){let n=M(e,`x`,`unique`,`string_or_numeric`);y(n.rank>0,()=>`The input tensor must be at least 1D`);let r={x:n},i={axis:t},[a,o]=j.runKernel(gr,r,i);return{values:a,indices:o}}var sm=N({unique_:om});function cm(e,t,n){let r=M(e,`x`,`unsortedSegmentSum`),i=M(t,`segmentIds`,`unsortedSegmentSum`,`int32`);y(ee(n),()=>`numSegments must be of dtype int`);let a={x:r,segmentIds:i},o={numSegments:n};return j.runKernel(vr,a,o)}var lm=N({unsortedSegmentSum_:cm});function um(e,t=0){let n=M(e,`x`,`unstack`,`string_or_numeric`);y(t>=-n.shape.length&&t<n.shape.length,()=>`Axis = ${t} is not in [-${n.shape.length}, ${n.shape.length})`);let r={value:n},i={axis:t};return j.runKernel(_r,r,i)}var dm=N({unstack_:um});function fm(e,t){return td(e,t,`right`)}function pm(e,t=!0,n,r){return j.makeVariable(e,t,n,r)}function mm(e,t){let n=[];for(let e=0;e<t.length;e++)t[e]&&n.push(e);let r=I(e,`int32`),i=I([n.length,e.length],`int32`);for(let t=0;t<n.length;t++){let a=r.indexToLoc(n[t]),o=t*e.length;i.values.set(a,o)}return i.toTensor()}async function hm(e){let t=M(e,`condition`,`whereAsync`,`bool`),n=await t.data(),r=mm(t.shape,n);return e!==t&&t.dispose(),r}var gm=hm;async function _m(e,t,n){let r=M(e,`tensor`,`boolMask`),i=M(t,`mask`,`boolMask`,`bool`),a=n??0,o=i.rank,s=r.shape;y(o>0,()=>`mask cannot be scalar`),b(s.slice(a,a+o),i.shape,`mask's shape must match the first K dimensions of tensor's shape,`);let c=1;for(let e=a;e<a+o;e++)c*=s[e];let l=V(r,s.slice(0,a).concat([c],s.slice(a+o))),u=V(i,[-1]),d=await gm(u),f=Ip(d,[1]),p=ou(l,f,a);return e!==r&&r.dispose(),t!==i&&i.dispose(),f.dispose(),l.dispose(),u.dispose(),d.dispose(),p}var vm=_m;function ym(e,t,n){let r=M(e,`x`,`transpose`);if(t??=r.shape.map((e,t)=>t).reverse(),y(r.rank===t.length,()=>`Error in transpose: rank of input ${r.rank} must match length of perm ${t}.`),t.forEach(e=>{y(e>=0&&e<r.rank,()=>`All entries in 'perm' must be between 0 and ${r.rank-1} but got ${t}`)}),r.rank<=1)return r.clone();let i={x:r},a={perm:t};return r.dtype===`complex64`?P(()=>{let e=Pf(r),t=fu(r);return e=j.runKernel(hr,{x:e},a),t=j.runKernel(hr,{x:t},a),n&&(t=Fu(t)),oa(e,t)}):j.runKernel(hr,i,a)}var bm=N({transpose_:ym});function xm(e,t,n,r,i=!0){let a=M(e,`v`,`movingAverage`),o=M(t,`x`,`movingAverage`),s=M(n,`decay`,`movingAverage`);zi(a,o),y(w(a.shape,o.shape),()=>`Shape mismatch in v and x`);let c=G(1),l=q(c,s),u=B(q(o,a),l);if(i){y(r!=null,()=>`When using zeroDebias: true, step is required.`);let e=M(r,`step`,`movingAverage`);u=z(u,q(c,Il(s,e)))}return R(a,u)}var Sm=N({movingAverage_:xm});function Cm(e,t,n){Se(n);let r=M(e,`indices`,`scatterND`,`int32`),i=M(t,`updates`,`scatterND`);Qp(i,r,n);let a={indices:r,updates:i},o={shape:n};return j.runKernel(zn,a,o)}var wm=N({scatterND_:Cm});function Tm(e,t,n,r){if(e.dtype!==`int32`)throw Error(`tf.sparseToDense() expects the indices to be int32 type, but the dtype was ${e.dtype}.`);if(e.rank>2)throw Error(`sparseIndices should be a scalar, vector, or matrix, but got shape ${e.shape}.`);let i=e.rank>0?e.shape[0]:1,a=e.rank>1?e.shape[1]:1;if(n.length!==a)throw Error(`outputShape has incorrect number of elements:, ${n.length}, should be: ${a}.`);let o=t.size;if(!(t.rank===0||t.rank===1&&o===i))throw Error(`sparseValues has incorrect shape ${t.shape}, should be [] or [${i}]`);if(t.dtype!==r.dtype)throw Error(`sparseValues.dtype must match defaultValues.dtype`)}function Em(e,t,n,r=0){Se(n);let i=M(e,`sparseIndices`,`sparseToDense`,`int32`),a=M(t,`sparseValues`,`sparseToDense`,`string_or_numeric`),o=M(r,`defaultValue`,`sparseToDense`,a.dtype);Tm(i,a,n,o);let s={sparseIndices:i,sparseValues:a,defaultValue:o},c={outputShape:n};return j.runKernel(rr,s,c)}var Dm=N({sparseToDense_:Em});function Om(e,t){let n=M(t,`indices`,`gatherND`,`int32`),r={params:M(e,`x`,`gatherND`,`string_or_numeric`),indices:n};return j.runKernel(Lt,r)}var km=N({gatherND_:Om});function Am(e,t){if(t==null)return e.shape.slice();if(w(e.shape,t))return t;if(e.shape.length===t.length){let n=[];for(let r=0;r<e.shape.length;r++)t[r]==null&&e.shape[r]!=null?n.push(e.shape[r]):n.push(t[r]);return n}return t}function jm(e,t,n,r){let i=M(e,`x`,`dropout`);if(y(i.dtype===`float32`,()=>`x has to be a floating point tensor since it's going to be scaled, but got a ${i.dtype} tensor instead.`),y(t>=0&&t<1,()=>`rate must be a float in the range [0, 1), but got ${t}.`),t===0)return e instanceof Ei?i.clone():i;let a=Am(i,n),o=1-t;return B(i,z(iu(R(kf(a,0,1,`float32`,r),o)),o))}var Mm=N({dropout_:jm});function Nm(e){return Math.floor(2**Math.ceil(Math.log(e)/Math.log(2)))}function Pm(e,t,n){let r=1-e%2,i=new Float32Array(e);for(let a=0;a<e;++a){let o=2*Math.PI*a/(e+r-1);i[a]=t-n*Math.cos(o)}return Gp(i,`float32`)}async function Fm(e,t,n=1){let r=M(e,`predictions`,`inTopK`),i=M(t,`targets`,`inTopK`);y(r.rank>1,()=>`inTopK() expects the predictions to be of rank 2 or higher, but got ${r.rank}`),y(r.rank-1===i.rank,()=>`predictions rank should be 1 larger than targets rank, but got predictions rank ${r.rank} and targets rank ${i.rank}`),b(r.shape.slice(0,r.shape.length-1),i.shape,`predictions's shape should be align with the targets' shape, except the last dimension.`);let a=r.shape[r.shape.length-1];y(n>0&&n<=a,()=>`'k' passed to inTopK() must be > 0 && <= the predictions last dimension (${a}), but got ${n}`);let o=await r.data(),s=await i.data(),[c,l]=[o.length/a,a],u=ie(`bool`,c);for(let e=0;e<c;e++){let t=e*l,r=o.subarray(t,t+l),i=[];for(let e=0;e<r.length;e++)i.push({value:r[e],index:e});i.sort((e,t)=>t.value-e.value),u[e]=0;for(let t=0;t<n;t++)if(i[t].index===s[e]){u[e]=1;break}}return e!==r&&r.dispose(),t!==i&&i.dispose(),ca(u,i.shape,`bool`)}var Im=Fm;function Lm(e,t,n,r,i,a=`NHWC`,o){let s=e;e.rank===3&&(s=V(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let c=t;c.rank===3&&(c=V(t,[1,t.shape[0],t.shape[1],t.shape[2]])),y(s.rank===4,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${s.shape}.`),y(c.rank===4,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${c.shape}.`),y(n.length===4,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${n}.`);let l=a===`NHWC`?s.shape[3]:s.shape[1],u=a===`NHWC`?c.shape[3]:c.shape[1];y(l===n[2],()=>`Error in conv2dDerFilter: depth of input ${l}) must match input depth in filter (${n[2]}.`),y(u===n[3],()=>`Error in conv2dDerFilter: depth of dy (${u}) must match output depth for filter (${n[3]}).`),js(`conv2dDerFilter`,i,o);let d={x:s,dy:c},f={strides:r,pad:i,dataFormat:a,dimRoundingMode:o,filterShape:n};return j.runKernel(lt,d,f)}var Rm=N({conv2DBackpropFilter_:Lm});function zm(e,t,n){if(n==null||n===`linear`)return e;if(n===`relu`)return B(e,Bp(t));throw Error(`Cannot compute gradient for fused activation ${n}.`)}function Bm(e,t){let n=t,r=il(e.shape,t.shape);return r.length>0&&(n=K(n,r)),V(n,e.shape)}function Vm(e,t,n,r){if(t===`linear`)return e;if(t===`relu`)return Rf(e);if(t===`elu`)return vl(e);if(t===`relu6`)return Bf(e);if(t===`prelu`)return $d(e,n);if(t===`leakyrelu`)return bu(e,r);if(t===`sigmoid`)return Vs(e);throw Error(`Unknown fused activation ${t}.`)}var Hm=(e,t)=>!(e>0)||t===`linear`;function Um({x:e,filter:t,strides:n,pad:r,dataFormat:i=`NHWC`,dilations:a=[1,1],dimRoundingMode:o,bias:s,activation:c=`linear`,preluActivationWeights:l,leakyreluAlpha:u}){if(c||=`linear`,Hm(j.state.gradientDepth,c)===!1){y(i===`NHWC`,()=>`Error in fused conv2d: got dataFormat of ${i} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let d=Ec(e,t,n,r,i,a,o);return s!=null&&(d=R(d,s)),Vm(d,c,l,u)}let d=M(e,`x`,`conv2d`,`float32`),f=M(t,`filter`,`conv2d`,`float32`),p=d,m=!1;d.rank===3&&(m=!0,p=V(d,[1,d.shape[0],d.shape[1],d.shape[2]])),y(p.rank===4,()=>`Error in fused conv2d: input must be rank 4, but got rank ${p.rank}.`),y(f.rank===4,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${f.rank}.`),js(`fused conv2d`,r,o);let h=i===`NHWC`?p.shape[3]:p.shape[1];y(f.shape[2]===h,()=>`Error in conv2d: depth of input (${h}) must match input depth for filter ${f.shape[2]}.`),y(Os(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`);let g=gs(p.shape,f.shape,n,a,r,o),_;s!=null&&(_=M(s,`bias`,`fused conv2d`),[_]=A(_,d),i===`NHWC`?W(g.outShape,_.shape):(y(_.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${_.shape.length}.`),y(_.shape.length===0||_.shape[0]===g.outChannels||_.shape[0]===1,()=>`Error in fused conv2d: bias shape (${_.shape}) is not compatible with the number of output channels (${g.outChannels})`)));let v;if(l!=null){let e=l.shape;if(y(e.length<=1||e.length===3,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${e.length}.`),e.length===1)y(e[0]===1||e[0]===g.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${e}) is not compatible with the number of output channels (${g.outChannels}).`);else if(e.length===3)try{W(e,g.outShape)}catch{let t=`Error in fused conv2d: PReLU activation weights (${e}) is not compatible with the output shape of the conv2d (${g.outShape}).`;throw Error(t)}v=M(l,`prelu weights`,`fused conv2d`)}let b=(e,t)=>{y(i===`NHWC`,()=>`Error in gradient of fused conv2D: got dataFormat of ${i} but only NHWC is currently supported.`);let[o,s,l,u]=t,d=zm(e,l,c);y(Ds(a),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${a}'`);let f=[Ac(s.shape,d,o,n,r),Rm(s,d,o.shape,n,r)];if(u!=null){let e=Bm(u,d);f.push(e)}return f},x={x:p,filter:f,bias:_,preluActivationWeights:v},S={strides:n,pad:r,dataFormat:i,dilations:a,dimRoundingMode:o,activation:c,leakyreluAlpha:u};return s==null?Nu((e,t,n)=>{let r=j.runKernel(wr,x,S);return n([t,e,r]),m&&(r=V(r,[r.shape[1],r.shape[2],r.shape[3]])),{value:r,gradFunc:b}})(p,f):Nu((e,t,n,r)=>{let i=j.runKernel(wr,x,S);return r([t,e,i,n]),m&&(i=V(i,[i.shape[1],i.shape[2],i.shape[3]])),{value:i,gradFunc:b}})(p,f,_)}var Wm=N({fusedConv2d_:Um});function Gm(e,t,n,r,i,a=[1,1],o){let s=e;e.rank===3&&(s=V(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let c=t;c.rank===3&&(c=V(t,[1,t.shape[0],t.shape[1],t.shape[2]]));let l={x:s,dy:c},u={strides:r,pad:i,dimRoundingMode:o,dilations:a,filterShape:n};return j.runKernel(xt,l,u)}var Km=N({depthwiseConv2dNativeBackpropFilter_:Gm});function qm(e,t,n,r,i,a=[1,1],o){let s=t,c=!1;t.rank===3&&(c=!0,s=V(t,[1,t.shape[0],t.shape[1],t.shape[2]]));let l={dy:s,filter:n},u={strides:r,pad:i,dimRoundingMode:o,dilations:a,inputShape:e},d=j.runKernel(St,l,u);return c?V(d,[d.shape[1],d.shape[2],d.shape[3]]):d}var Jm=N({depthwiseConv2dNativeBackpropInput_:qm});function Ym({x:e,filter:t,strides:n,pad:r,dataFormat:i=`NHWC`,dilations:a=[1,1],dimRoundingMode:o,bias:s,activation:c=`linear`,preluActivationWeights:l,leakyreluAlpha:u}){if(Hm(j.state.gradientDepth,c)===!1){let d=Qc(e,t,n,r,i,a,o);return s!=null&&(d=R(d,s)),Vm(d,c,l,u)}let d=M(e,`x`,`depthwiseConv2d`,`float32`),f=M(t,`filter`,`depthwiseConv2d`,`float32`),p=d,m=!1;d.rank===3&&(m=!0,p=V(d,[1,d.shape[0],d.shape[1],d.shape[2]])),y(p.rank===4,()=>`Error in fused depthwiseConv2d: input must be rank 4, but got rank ${p.rank}.`),y(f.rank===4,()=>`Error in fused depthwiseConv2d: filter must be rank 4, but got rank ${f.rank}.`),y(p.shape[3]===f.shape[2],()=>`Error in fused depthwiseConv2d: number of input channels (${p.shape[3]}) must match the inChannels dimension in filter ${f.shape[2]}.`),a??=[1,1],y(Os(n,a),()=>`Error in fused depthwiseConv2d: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),js(`fused depthwiseConv2d`,r,o);let h=gs(p.shape,f.shape,n,a,r,o,!0),g;s!=null&&(g=M(s,`bias`,`fused conv2d`),[g]=A(g,d),W(h.outShape,g.shape));let _;l!=null&&(_=M(l,`prelu weights`,`fused depthwiseConv2d`));let v=(e,t)=>{y(Ds(a),()=>`Error in gradient of fused depthwiseConv2d: dilation rates greater than 1 are not yet supported. Got dilations '${a}'`);let[i,s,l,u]=t,d=zm(e,l,c),f=Jm(s.shape,d,i,n,r,a,o),p=Km(s,d,i.shape,n,r,a,o);return u==null?[f,p]:[f,p,Bm(g,d)]},b={x:p,filter:f,bias:g,preluActivationWeights:_},x={strides:n,pad:r,dataFormat:i,dilations:a,dimRoundingMode:o,activation:c,leakyreluAlpha:u};return s==null?Nu((e,t,n)=>{let r=j.runKernel(Tr,b,x);return n([t,e,r]),m&&(r=V(r,[r.shape[1],r.shape[2],r.shape[3]])),{value:r,gradFunc:v}})(p,f):Nu((e,t,n,r)=>{let i=j.runKernel(Tr,b,x);return r([t,e,i,n]),m&&(i=V(i,[i.shape[1],i.shape[2],i.shape[3]])),{value:i,gradFunc:v}})(p,f,g)}var Xm=N({fusedDepthwiseConv2d_:Ym});function Zm({a:e,b:t,transposeA:n=!1,transposeB:r=!1,bias:i,activation:a=`linear`,preluActivationWeights:o,leakyreluAlpha:s=.2}){if(Hm(j.state.gradientDepth,a)===!1){let c=H(e,t,n,r);return i!=null&&(c=R(c,i)),Vm(c,a,o,s)}let c=M(e,`a`,`fused matMul`),l=M(t,`b`,`fused matMul`);[c,l]=A(c,l);let u=n?c.shape[c.rank-2]:c.shape[c.rank-1],d=r?l.shape[l.rank-1]:l.shape[l.rank-2],f=n?c.shape[c.rank-1]:c.shape[c.rank-2],p=r?l.shape[l.rank-2]:l.shape[l.rank-1],m=c.shape.slice(0,-2),h=l.shape.slice(0,-2),g=S(m),_=S(h);y(u===d,()=>`Error in fused matMul: inner shapes (${u}) and (${d}) of Tensors with shapes ${c.shape} and ${l.shape} and transposeA=${n} and transposeB=${r} must match.`);let v=W(c.shape.slice(0,-2),l.shape.slice(0,-2)).concat([f,p]),b=n?V(c,[g,u,f]):V(c,[g,f,u]),x=r?V(l,[_,p,d]):V(l,[_,d,p]),C;i!=null&&(C=M(i,`bias`,`fused matMul`),[C]=A(C,c),W(v,C.shape));let w;o!=null&&(w=M(o,`prelu weights`,`fused matMul`));let ee=(e,t)=>{let[o,s,c,l]=t,u=zm(V(e,c.shape),c,a),d,f;if(!n&&!r?(d=H(u,s,!1,!0),f=H(o,u,!0,!1)):!n&&r?(d=H(u,s,!1,!1),f=H(u,o,!0,!1)):n&&!r?(d=H(s,u,!1,!0),f=H(o,u,!1,!1)):(d=H(s,u,!0,!0),f=H(u,o,!0,!0)),i!=null){let e=Bm(l,u);return[d,f,e]}else return[d,f]},T={a:b,b:x,bias:C,preluActivationWeights:w},E={transposeA:n,transposeB:r,activation:a,leakyreluAlpha:s};return i==null?Nu((e,t,n)=>{let r=j.runKernel(Cr,T,E);return n([e,t,r]),{value:V(r,v),gradFunc:ee}})(b,x):Nu((e,t,n,r)=>{let i=j.runKernel(Cr,T,E);return r([e,t,i,n]),{value:V(i,v),gradFunc:ee}})(b,x,C)}var Qm=N({fusedMatMul_:Zm}),$m=s({conv2d:()=>Wm,depthwiseConv2d:()=>Xm,matMul:()=>Qm});function eh(e){return Pm(e,.54,.46)}var th=N({hammingWindow_:eh});function nh(e){return Pm(e,.5,.5)}var rh=N({hannWindow_:nh});function ih(e,t,n,r=!1,i=0){let a=0,o=[];for(;a+t<=e.size;)o.push(U(e,a,t)),a+=n;if(r)for(;a<e.size;){let r=a+t-e.size,s=Rs([U(e,a,t-r),mc([r],i)]);o.push(s),a+=n}return o.length===0?Kp([],[0,t]):V(Rs(o),[o.length,t])}var ah=N({frame_:ih});function oh(e,t,n,r,i=rh){return r??=Nm(t),Mp(B(ah(e,t,n),i(t)),r)}var sh=N({stft_:oh});function ch(e,t,n,r,i=`bilinear`,a=0){let o=M(e,`image`,`cropAndResize`),s=M(t,`boxes`,`cropAndResize`,`float32`),c=M(n,`boxInd`,`cropAndResize`,`int32`),l=s.shape[0];y(o.rank===4,()=>`Error in cropAndResize: image must be rank 4,but got rank ${o.rank}.`),y(s.rank===2&&s.shape[1]===4,()=>`Error in cropAndResize: boxes must be have size [${l},4] but had shape ${s.shape}.`),y(c.rank===1&&c.shape[0]===l,()=>`Error in cropAndResize: boxInd must be have size [${l}] but had shape ${s.shape}.`),y(r.length===2,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${r.length}.`),y(r[0]>=1&&r[1]>=1,()=>`cropSize must be atleast [1,1], but was ${r}`),y(i===`bilinear`||i===`nearest`,()=>`method must be bilinear or nearest, but was ${i}`);let u={image:o,boxes:s,boxInd:c},d={method:i,extrapolationValue:a,cropSize:r};return j.runKernel(_t,u,d)}var lh=N({cropAndResize_:ch});function uh(e){let t=M(e,`image`,`flipLeftRight`,`float32`);y(t.rank===4,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${t.rank}.`);let n={image:t};return j.runKernel(Mt,n,{})}var dh=N({flipLeftRight_:uh});function fh(e){let t=M(e,`image`,`grayscaleToRGB`),n=t.rank-1,r=t.shape[n];y(t.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${t.rank}.`),y(r===1,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${r}.`);let i=Array(t.rank);return i.fill(1,0,n),i[n]=3,eu(t,i)}var ph=N({grayscaleToRGB_:fh});function mh(e){let t=M(e,`image`,`RGBToGrayscale`),n=t.rank-1,r=t.shape[n];y(t.rank>=2,()=>`Error in RGBToGrayscale: images must be at least rank 2, but got rank ${t.rank}.`),y(r===3,()=>`Error in RGBToGrayscale: last dimension of an RGB image should be size 3, but got size ${r}.`);let i=t.dtype,a=L(t,`float32`),o=Gp([.2989,.587,.114]),s;switch(t.rank){case 2:s=gl(`ij,j->i`,a,o);break;case 3:s=gl(`ijk,k->ij`,a,o);break;case 4:s=gl(`ijkl,l->ijk`,a,o);break;case 5:s=gl(`ijklm,m->ijkl`,a,o);break;case 6:s=gl(`ijklmn,n->ijklm`,a,o);break;default:throw Error(`Not a valid tensor rank.`)}return s=Xl(s,-1),L(s,i)}var hh=N({rgbToGrayscale_:mh});function gh(e,t,n=0,r=.5){let i=M(e,`image`,`rotateWithOffset`,`float32`);y(i.rank===4,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${i.rank}.`);let a={image:i},o={radians:t,fillValue:n,center:r};return j.runKernel(Sr,a,o)}var _h=N({rotateWithOffset_:gh});function vh(e,t,n,r,i,a){r??=.5,i??=-1/0,a??=0;let o=e.shape[0];return n=Math.min(n,o),y(0<=r&&r<=1,()=>`iouThreshold must be in [0, 1], but was '${r}'`),y(e.rank===2,()=>`boxes must be a 2D tensor, but was of rank '${e.rank}'`),y(e.shape[1]===4,()=>`boxes must have 4 columns, but 2nd dimension was ${e.shape[1]}`),y(t.rank===1,()=>`scores must be a 1D tensor`),y(t.shape[0]===o,()=>`scores has incompatible shape with boxes. Expected ${o}, but was ${t.shape[0]}`),y(0<=a&&a<=1,()=>`softNmsSigma must be in [0, 1], but was '${a}'`),{maxOutputSize:n,iouThreshold:r,scoreThreshold:i,softNmsSigma:a}}function yh(e,t,n,r=.5,i=-1/0){let a=M(e,`boxes`,`nonMaxSuppression`,`float32`),o=M(t,`scores`,`nonMaxSuppression`,`float32`),s=vh(a,o,n,r,i);n=s.maxOutputSize,r=s.iouThreshold,i=s.scoreThreshold;let c={maxOutputSize:n,iouThreshold:r,scoreThreshold:i};return j.runKernel(mn,{boxes:a,scores:o},c)}var bh=N({nonMaxSuppression_:yh});function xh(e,t,n){let r=Sh(e,t,n),i=r<0?-(r+1):r;e.splice(i,0,t)}function Sh(e,t,n){return wh(e,t,n||Ch)}function Ch(e,t){return e>t?1:e<t?-1:0}function wh(e,t,n){let r=0,i=e.length,a=0,o=!1;for(;r<i;){a=r+(i-r>>>1);let s=n(t,e[a]);s>0?r=a+1:(i=a,o=!s)}return o?r:-r-1}function Th(e,t,n,r,i){return Oh(e,t,n,r,i,0)}function Eh(e,t,n,r,i,a){return Oh(e,t,n,r,i,0,!1,a,!0)}function Dh(e,t,n,r,i,a){return Oh(e,t,n,r,i,a,!0)}function Oh(e,t,n,r,i,a,o=!1,s=!1,c=!1){let l=[];for(let e=0;e<t.length;e++)t[e]>i&&l.push({score:t[e],boxIndex:e,suppressBeginIndex:0});l.sort(jh);let u=a>0?-.5/a:0,d=[],f=[];for(;d.length<n&&l.length>0;){let t=l.pop(),{score:n,boxIndex:a,suppressBeginIndex:o}=t;if(n<i)break;let s=!1;for(let n=d.length-1;n>=o;--n){let o=kh(e,a,d[n]);if(o>=r){s=!0;break}if(t.score*=Ah(r,u,o),t.score<=i)break}t.suppressBeginIndex=d.length,s||(t.score===n?(d.push(a),f.push(t.score)):t.score>i&&xh(l,t,jh))}let p=d.length,m=n-p;s&&m>0&&(d.push(...Array(m).fill(0)),f.push(...Array(m).fill(0)));let h={selectedIndices:d};return o&&(h.selectedScores=f),c&&(h.validOutputs=p),h}function kh(e,t,n){let r=e.subarray(t*4,t*4+4),i=e.subarray(n*4,n*4+4),a=Math.min(r[0],r[2]),o=Math.min(r[1],r[3]),s=Math.max(r[0],r[2]),c=Math.max(r[1],r[3]),l=Math.min(i[0],i[2]),u=Math.min(i[1],i[3]),d=Math.max(i[0],i[2]),f=Math.max(i[1],i[3]),p=(s-a)*(c-o),m=(d-l)*(f-u);if(p<=0||m<=0)return 0;let h=Math.max(a,l),g=Math.max(o,u),_=Math.min(s,d),v=Math.min(c,f),y=Math.max(_-h,0)*Math.max(v-g,0);return y/(p+m-y)}function Ah(e,t,n){let r=Math.exp(t*n*n);return n<=e?r:0}function jh(e,t){return e.score-t.score||e.score===t.score&&t.boxIndex-e.boxIndex}async function Mh(e,t,n,r=.5,i=-1/0){let a=M(e,`boxes`,`nonMaxSuppressionAsync`),o=M(t,`scores`,`nonMaxSuppressionAsync`),s=vh(a,o,n,r,i);n=s.maxOutputSize,r=s.iouThreshold,i=s.scoreThreshold;let c=await Promise.all([a.data(),o.data()]),l=c[0],u=c[1],{selectedIndices:d}=Th(l,u,n,r,i);return a!==e&&a.dispose(),o!==t&&o.dispose(),Gp(d,`int32`)}var Nh=Mh;function Ph(e,t,n,r=.5,i=-1/0,a=0){let o=M(e,`boxes`,`nonMaxSuppression`),s=M(t,`scores`,`nonMaxSuppression`),c=vh(o,s,n,r,i,a);n=c.maxOutputSize,r=c.iouThreshold,i=c.scoreThreshold,a=c.softNmsSigma;let l={boxes:o,scores:s},u={maxOutputSize:n,iouThreshold:r,scoreThreshold:i,softNmsSigma:a},d=j.runKernel(gn,l,u);return{selectedIndices:d[0],selectedScores:d[1]}}var Fh=N({nonMaxSuppressionWithScore_:Ph});async function Ih(e,t,n,r=.5,i=-1/0,a=0){let o=M(e,`boxes`,`nonMaxSuppressionAsync`),s=M(t,`scores`,`nonMaxSuppressionAsync`),c=vh(o,s,n,r,i,a);n=c.maxOutputSize,r=c.iouThreshold,i=c.scoreThreshold,a=c.softNmsSigma;let l=await Promise.all([o.data(),s.data()]),u=l[0],d=l[1],{selectedIndices:f,selectedScores:p}=Dh(u,d,n,r,i,a);return o!==e&&o.dispose(),s!==t&&s.dispose(),{selectedIndices:Gp(f,`int32`),selectedScores:Gp(p)}}var Lh=Ih;function Rh(e,t,n,r=.5,i=-1/0,a=!1){let o=M(e,`boxes`,`nonMaxSuppression`),s=M(t,`scores`,`nonMaxSuppression`),c=vh(o,s,n,r,i,null),l=c.maxOutputSize,u=c.iouThreshold,d=c.scoreThreshold,f={boxes:o,scores:s},p={maxOutputSize:l,iouThreshold:u,scoreThreshold:d,padToMaxOutputSize:a},m=j.runKernel(hn,f,p);return{selectedIndices:m[0],validOutputs:m[1]}}var zh=N({nonMaxSuppressionPadded_:Rh});async function Bh(e,t,n,r=.5,i=-1/0,a=!1){let o=M(e,`boxes`,`nonMaxSuppressionAsync`),s=M(t,`scores`,`nonMaxSuppressionAsync`),c=vh(o,s,n,r,i,null),l=c.maxOutputSize,u=c.iouThreshold,d=c.scoreThreshold,[f,p]=await Promise.all([o.data(),s.data()]),{selectedIndices:m,validOutputs:h}=Eh(f,p,l,u,d,a);return o!==e&&o.dispose(),s!==t&&s.dispose(),{selectedIndices:Gp(m,`int32`),validOutputs:G(h,`int32`)}}var Vh=Bh;function Hh(e,t,n=!1,r=!1){let i=M(e,`images`,`resizeBilinear`);y(i.rank===3||i.rank===4,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${i.rank}.`),y(t.length===2,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${t}.`),y(r===!1||n===!1,()=>`Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.`);let a=i,o=!1;i.rank===3&&(o=!0,a=V(i,[1,i.shape[0],i.shape[1],i.shape[2]]));let[]=t,s={images:a},c={alignCorners:n,halfPixelCenters:r,size:t},l=j.runKernel(Nn,s,c);return o?V(l,[l.shape[1],l.shape[2],l.shape[3]]):l}var Uh=N({resizeBilinear_:Hh});function Wh(e,t,n=!1,r=!1){let i=M(e,`images`,`resizeNearestNeighbor`);y(i.rank===3||i.rank===4,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${i.rank}.`),y(t.length===2,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${t}.`),y(i.dtype===`float32`||i.dtype===`int32`,()=>"`images` must have `int32` or `float32` as dtype"),y(r===!1||n===!1,()=>`Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.`);let a=i,o=!1;i.rank===3&&(o=!0,a=V(i,[1,i.shape[0],i.shape[1],i.shape[2]]));let[]=t,s={images:a},c={alignCorners:n,halfPixelCenters:r,size:t},l=j.runKernel(jn,s,c);return o?V(l,[l.shape[1],l.shape[2],l.shape[3]]):l}var Gh=N({resizeNearestNeighbor_:Wh});function Kh(e,t=`binary`,n=!1,r=.5){let i=M(e,`image`,`threshold`),a=i.shape[0]*i.shape[1],o=B(Gp([r]),255),s,c,l,u;if(y(i.rank===3,()=>`Error in threshold: image must be rank 3,but got rank ${i.rank}.`),y(i.shape[2]===3||i.shape[2]===1,()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${i.shape[2]}.`),y(i.dtype===`int32`||i.dtype===`float32`,()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${i.dtype}.`),y(t===`otsu`||t===`binary`,()=>`Method must be binary or otsu, but was ${t}`),i.shape[2]===3){[s,c,l]=Ap(i,[1,1,1],-1);let e=B(s,.2989),t=B(c,.587),n=B(l,.114);u=R(R(e,t),n)}else u=e;return t===`otsu`&&(o=qh(ac(L(Qf(u),`int32`),ca([]),256),a)),L(B(n?wu(u,o):cu(u,o),255),`int32`)}function qh(e,t){let n=Gp([-1]),r=Gp([0]),i=Gp([0]),a,o,s,c,l,u;for(let d=0;d<e.size-1;d++){a=U(e,0,d+1),o=U(e,d+1),l=z(K(a),t),u=z(K(o),t),s=z(K(B(a,Mf(0,a.size))),K(a));let f=mc(o.shape,a.size),p=R(Mf(0,o.size),f);c=z(K(B(o,p)),K(o));let m=q(s,c),h=q(s,c);i=B(B(B(l,u),m),h);let g=cu(i,r);r=cl(g,i,r),n=cl(g,Gp([d]),n)}return n}var Jh=N({threshold_:Kh});function Yh(e,t,n=`nearest`,r=`constant`,i=0,a){let o=M(e,`image`,`transform`,`float32`),s=M(t,`transforms`,`transform`,`float32`);y(o.rank===4,()=>`Error in transform: image must be rank 4,but got rank ${o.rank}.`),y(s.rank===2&&(s.shape[0]===o.shape[0]||s.shape[0]===1)&&s.shape[1]===8,()=>`Error in transform: Input transform should be batch x 8 or 1 x 8`),y(a==null||a.length===2,()=>`Error in transform: outputShape must be [height, width] or null, but got ${a}.`);let c={image:o,transforms:s},l={interpolation:n,fillMode:r,fillValue:i,outputShape:a};return j.runKernel(mr,c,l)}var Xh=N({transform_:Yh});function Zh(e,t,n){let r=M(e,`a`,`bandPart`);y(r.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${r.rank}.`);let i=r.shape,[a,o]=r.shape.slice(-2),s,c;typeof t==`number`?(y(t%1==0,()=>`bandPart(): numLower must be an integer, got ${t}.`),y(t<=a,()=>`bandPart(): numLower (${t}) must not be greater than the number of rows (${a}).`),s=M(t<0?a:t,`numLower`,`bandPart`)):(y(t.dtype===`int32`,()=>`bandPart(): numLower's dtype must be an int32.`),s=cl(Su(t,0),a,_d(t,a))),typeof n==`number`?(y(n%1==0,()=>`bandPart(): numUpper must be an integer, got ${n}.`),y(n<=o,()=>`bandPart(): numUpper (${n}) must not be greater than the number of columns (${o}).`),c=M(n<0?o:n,`numUpper`,`bandPart`)):(y(n.dtype===`int32`,()=>`bandPart(): numUpper's dtype must be an int32.`),c=cl(Su(n,0),o,_d(n,o)));let l=q(V(Mf(0,a,1,`int32`),[-1,1]),Mf(0,o,1,`int32`)),u=Ku(wu(l,s),uu(l,Fu(c))),d=pd([a,o],r.dtype);return V(Rp(dm(V(r,[-1,a,o])).map(e=>cl(u,e,d))),i)}var Qh=N({bandPart_:Zh});function $h(e){let t;if(Array.isArray(e)){t=!1,y(e!=null&&e.length>0,()=>`Gram-Schmidt process: input must not be null, undefined, or empty`);let n=e[0].shape[0];for(let t=1;t<e.length;++t)y(e[t].shape[0]===n,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${e[t].shape[0]} vs. ${n})`)}else t=!0,e=Ap(e,e.shape[0],0).map(e=>Ip(e,[0]));y(e.length<=e[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${e.length}) exceeds number of dimensions (${e[0].shape[0]}).`);let n=[],r=e;for(let t=0;t<e.length;++t)n.push(j.tidy(()=>{let e=r[t];if(t>0)for(let r=0;r<t;++r){let t=B(K(B(n[r],e)),n[r]);e=q(e,t)}return z(e,Wl(e,`euclidean`))}));return t?Rp(n,0):n}var eg=N({gramSchmidt_:$h});function tg(e,t=!1){if(y(e.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${e.rank}`),e.rank===2)return ng(e,t);{let n=dm(V(e,[e.shape.slice(0,e.shape.length-2).reduce((e,t)=>e*t),e.shape[e.shape.length-2],e.shape[e.shape.length-1]]),0),r=[],i=[];return n.forEach(e=>{let[n,a]=ng(e,t);r.push(n),i.push(a)}),[V(Rp(r,0),e.shape),V(Rp(i,0),e.shape)]}}function ng(e,t=!1){return j.tidy(()=>{y(e.shape.length===2,()=>`qr2d() requires a 2D Tensor, but got a ${e.shape.length}D Tensor.`);let n=e.shape[0],r=e.shape[1],i=nu(n),a=Po(e),o=Kp([[1]],[1,1]),s=Po(o),c=n>=r?r:n;for(let e=0;e<c;++e){let t=a,c=s,l=i;[s,a,i]=j.tidy(()=>{let t=U(a,[e,e],[n-e,1]),c=Wl(t),l=U(a,[e,e],[1,1]),u=cl(cu(l,0),Kp([[-1]]),Kp([[1]])),d=q(l,B(u,c)),f=z(t,d);s=f.shape[0]===1?Po(o):Rs([o,U(f,[1,0],[f.shape[0]-1,f.shape[1]])],0);let p=Fu(z(H(u,d),c)),m=U(a,[e,0],[n-e,r]),h=B(p,s),g=bm(s);if(e===0)a=q(m,H(h,H(g,m)));else{let t=q(m,H(h,H(g,m)));a=Rs([U(a,[0,0],[e,r]),t],0)}let _=bm(h),v=U(i,[0,e],[n,i.shape[1]-e]);if(e===0)i=q(v,H(H(v,s),_));else{let t=q(v,H(H(v,s),_));i=Rs([U(i,[0,0],[n,e]),t],1)}return[s,a,i]}),F([t,c,l])}return!t&&n>r&&(i=U(i,[0,0],[n,r]),a=U(a,[0,0],[r,r])),[i,a]})}var rg=N({qr_:tg}),ig;(function(e){e[e.NONE=0]=`NONE`,e[e.MEAN=1]=`MEAN`,e[e.SUM=2]=`SUM`,e[e.SUM_BY_NONZERO_WEIGHTS=3]=`SUM_BY_NONZERO_WEIGHTS`})(ig||={});function ag(e,t,n=ig.SUM_BY_NONZERO_WEIGHTS){let r=M(e,`losses`,`computeWeightedLoss`),i=null;t!=null&&(i=M(t,`weights`,`computeWeightedLoss`));let a=i==null?r:B(r,i);if(n===ig.NONE)return a;if(n===ig.SUM)return K(a);if(n===ig.MEAN){if(i==null)return fd(a);{let e=r.size/i.size,t=z(K(a),K(i));return e>1?z(t,G(e)):t}}if(n===ig.SUM_BY_NONZERO_WEIGHTS){if(i==null)return z(K(a),G(r.size));{let e=L(K(kd(B(i,md(r.shape)),G(0))),`float32`);return z(K(a),e)}}throw Error(`Unknown reduction: ${n}`)}var og=N({computeWeightedLoss_:ag});function sg(e,t,n,r=ig.SUM_BY_NONZERO_WEIGHTS){let i=M(e,`labels`,`absoluteDifference`),a=M(t,`predictions`,`absoluteDifference`),o=null;return n!=null&&(o=M(n,`weights`,`absoluteDifference`)),b(i.shape,a.shape,`Error in absoluteDifference: `),og(Ho(q(i,a)),o,r)}var cg=N({absoluteDifference_:sg});function lg(e,t,n,r,i=ig.SUM_BY_NONZERO_WEIGHTS){let a=M(e,`labels`,`cosineDistance`),o=M(t,`predictions`,`cosineDistance`),s=null;return r!=null&&(s=M(r,`weights`,`cosineDistance`)),b(a.shape,o.shape,`Error in cosineDistance: `),og(q(G(1),K(B(a,o),n,!0)),s,i)}var ug=N({cosineDistance_:lg});function dg(e,t,n,r=ig.SUM_BY_NONZERO_WEIGHTS){let i=M(e,`labels`,`hingeLoss`),a=M(t,`predictions`,`hingeLoss`),o=null;n!=null&&(o=M(n,`weights`,`hingeLoss`)),b(i.shape,a.shape,`Error in hingeLoss: `);let s=G(1);return i=q(B(G(2),i),s),og(Rf(q(s,B(i,a))),o,r)}var fg=N({hingeLoss_:dg});function pg(e,t,n,r=1,i=ig.SUM_BY_NONZERO_WEIGHTS){let a=M(e,`labels`,`huberLoss`),o=M(t,`predictions`,`huberLoss`),s=null;n!=null&&(s=M(n,`weights`,`huberLoss`)),b(a.shape,o.shape,`Error in huberLoss: `);let c=G(r),l=Ho(q(o,a)),u=_d(l,c),d=q(l,u);return og(R(B(G(.5),Bl(u)),B(c,d)),s,i)}var mg=N({huberLoss_:pg});function hg(e,t,n,r=1e-7,i=ig.SUM_BY_NONZERO_WEIGHTS){let a=M(e,`labels`,`logLoss`),o=M(t,`predictions`,`logLoss`),s=null;n!=null&&(s=M(n,`weights`,`logLoss`)),b(a.shape,o.shape,`Error in logLoss: `);let c=G(1),l=G(r);return og(q(Fu(B(a,ku(R(o,l)))),B(q(c,a),ku(R(q(c,o),l)))),s,i)}var gg=N({logLoss_:hg});function _g(e,t,n,r=ig.SUM_BY_NONZERO_WEIGHTS){let i=M(e,`labels`,`meanSquaredError`),a=M(t,`predictions`,`meanSquaredError`),o=null;return n!=null&&(o=M(n,`weights`,`meanSquaredError`)),b(i.shape,a.shape,`Error in meanSquaredError: `),og(Pp(i,a),o,r)}var vg=N({meanSquaredError_:_g});function yg(e,t){let n=M(e,`labels`,`sigmoidCrossEntropyWithLogits`),r=M(t,`logits`,`sigmoidCrossEntropyWithLogits`);b(n.shape,r.shape,`Error in sigmoidCrossEntropyWithLogits: `);let i=Rf(r),a=B(r,n),o=ju(Jl(Fu(Ho(r))));return R(q(i,a),o)}function bg(e,t,n,r=0,i=ig.SUM_BY_NONZERO_WEIGHTS){let a=M(e,`multiClassLabels`,`sigmoidCrossEntropy`),o=M(t,`logits`,`sigmoidCrossEntropy`),s=null;if(n!=null&&(s=M(n,`weights`,`sigmoidCrossEntropy`)),b(a.shape,o.shape,`Error in sigmoidCrossEntropy: `),r>0){let e=G(r),t=G(1),n=G(.5);a=R(B(a,q(t,e)),B(n,e))}return og(yg(a,o),s,i)}var xg=N({sigmoidCrossEntropy_:bg});function Sg(e,t,n=-1){if(n===-1&&(n=t.rank-1),n!==t.rank-1)throw Error(`Softmax cross entropy along a non-last dimension is not yet supported. Labels / logits was rank ${t.rank} and dim was ${n}`);return Nu((e,t,r)=>{let i=Wu(t,[n],!0),a=q(L(t,`float32`),i);return r([e,a]),{value:K(Fu(B(a,e)),[n]),gradFunc:(e,t)=>{let[r,i]=t,a=El(e.shape,[n]);return[B(V(e,a),q(L(r,`float32`),Jl(i))),B(V(e,a),q(Jl(i),L(r,`float32`)))]}}})(e,t)}function Cg(e,t,n,r=0,i=ig.SUM_BY_NONZERO_WEIGHTS){let a=M(e,`onehotLabels`,`softmaxCrossEntropy`),o=M(t,`logits`,`softmaxCrossEntropy`),s=null;if(n!=null&&(s=M(n,`weights`,`softmaxCrossEntropy`)),b(a.shape,o.shape,`Error in softmaxCrossEntropy: `),r>0){let e=G(r),t=G(1),n=G(a.shape[1]);a=R(B(a,q(t,e)),z(e,n))}return og(Sg(a,o),s,i)}var wg=N({softmaxCrossEntropy_:Cg});function Tg(e,t,n,r){let i=M(e,`indices`,`sparseFillEmptyRows`,`int32`),a=M(t,`values`,`sparseFillEmptyRows`),o=M(n,`denseShape`,`sparseFillEmptyRows`,`int32`),s=M(r,`defaultValue`,`sparseFillEmptyRows`,a.dtype);if(i.rank!==2)throw Error(`Indices should be Tensor2D but received shape
        ${i.shape}`);if(a.rank!==1)throw Error(`Values should be Tensor1D but received shape ${a.shape}`);if(o.rank!==1)throw Error(`Dense shape should be Tensor1D but received shape ${o.shape}`);if(s.rank!==0)throw Error(`Default value should be a scalar but received shape ${s.shape}`);let c={indices:i,values:a,denseShape:o,defaultValue:s},l=j.runKernel($n,c);return{outputIndices:l[0],outputValues:l[1],emptyRowIndicator:l[2],reverseIndexMap:l[3]}}var Eg=N({sparseFillEmptyRows_:Tg});function Dg(e,t,n){let r=M(e,`inputIndices`,`sparseReshape`,`int32`),i=M(t,`inputShape`,`sparseReshape`,`int32`),a=M(n,`newShape`,`sparseReshape`,`int32`);if(r.rank!==2)throw Error(`Input indices should be Tensor2D but received shape
        ${r.shape}`);if(i.rank!==1)throw Error(`Input shape should be Tensor1D but received shape ${i.shape}`);if(a.rank!==1)throw Error(`New shape should be Tensor1D but received shape ${a.shape}`);let o={inputIndices:r,inputShape:i,newShape:a},s=j.runKernel(er,o);return{outputIndices:s[0],outputShape:s[1]}}var Og=N({sparseReshape_:Dg});function kg(e,t,n){let r=M(e,`data`,`sparseSegmentMean`),i=M(t,`indices`,`sparseSegmentMean`,`int32`),a=M(n,`segmentIds`,`sparseSegmentMean`,`int32`);if(r.rank<1)throw Error(`Data should be at least 1 dimensional but received scalar`);if(i.rank!==1)throw Error(`Indices should be Tensor1D but received shape
          ${i.shape}`);if(a.rank!==1)throw Error(`Segment ids should be Tensor1D but received shape
          ${a.shape}`);let o={data:r,indices:i,segmentIds:a};return j.runKernel(tr,o)}var Ag=N({sparseSegmentMean_:kg});function jg(e,t,n){let r=M(e,`data`,`sparseSegmentSum`),i=M(t,`indices`,`sparseSegmentSum`,`int32`),a=M(n,`segmentIds`,`sparseSegmentSum`,`int32`);if(r.rank<1)throw Error(`Data should be at least 1 dimensional but received scalar`);if(i.rank!==1)throw Error(`Indices should be Tensor1D but received shape
         ${i.shape}`);if(a.rank!==1)throw Error(`Segment ids should be Tensor1D but received shape
         ${a.shape}`);let o={data:r,indices:i,segmentIds:a};return j.runKernel(nr,o)}var Mg=N({sparseSegmentSum_:jg});function Ng(e,t,n,r,i,a,o,s){let c=M(e,`data`,`stringNGrams`,`string`);if(c.dtype!==`string`)throw Error(`Data must be of datatype string`);if(c.shape.length!==1)throw Error(`Data must be a vector, saw: ${c.shape}`);let l=M(t,`dataSplits`,`stringNGrams`);if(l.dtype!==`int32`)throw Error(`Data splits must be of datatype int32`);let u={separator:n,nGramWidths:r,leftPad:i,rightPad:a,padWidth:o,preserveShortSequences:s},d={data:c,dataSplits:l},f=j.runKernel(cr,d,u);return{nGrams:f[0],nGramsSplits:f[1]}}var Pg=N({stringNGrams_:Ng});function Fg(e,t,n=!0){let r=M(e,`input`,`stringSplit`,`string`),i=M(t,`delimiter`,`stringSplit`,`string`);if(r.rank!==1)throw Error(`Input should be Tensor1D but received shape ${r.shape}`);if(i.rank!==0)throw Error(`Delimiter should be a scalar but received shape ${i.shape}`);let a={skipEmpty:n},o={input:r,delimiter:i},s=j.runKernel(lr,o,a);return{indices:s[0],values:s[1],shape:s[2]}}var Ig=N({stringSplit_:Fg});function Lg(e,t){let n=M(e,`input`,`stringToHashBucketFast`,`string`),r={numBuckets:t};if(t<=0)throw Error(`Number of buckets must be at least 1`);let i={input:n};return j.runKernel(ur,i,r)}var Rg=N({stringToHashBucketFast_:Lg});function zg(e,t,n,r=!0){let i=M(e,`input`,`staticRegexReplace`,`string`),a={pattern:t,rewrite:n,replaceGlobal:r};return j.runKernel(or,{x:i},a)}var Bg=N({staticRegexReplace_:zg}),Vg={fft:wp,ifft:Ep,rfft:Mp,irfft:Op},Hg={hammingWindow:th,hannWindow:rh,frame:ah,stft:sh},Ug={flipLeftRight:dh,grayscaleToRGB:ph,resizeNearestNeighbor:Gh,resizeBilinear:Uh,rgbToGrayscale:hh,rotateWithOffset:_h,cropAndResize:lh,nonMaxSuppression:bh,nonMaxSuppressionAsync:Nh,nonMaxSuppressionWithScore:Fh,nonMaxSuppressionWithScoreAsync:Lh,nonMaxSuppressionPadded:zh,nonMaxSuppressionPaddedAsync:Vh,threshold:Jh,transform:Xh},Wg={bandPart:Qh,gramSchmidt:eg,qr:rg},Gg={absoluteDifference:cg,computeWeightedLoss:og,cosineDistance:ug,hingeLoss:fg,huberLoss:mg,logLoss:gg,meanSquaredError:vg,sigmoidCrossEntropy:xg,softmaxCrossEntropy:wg},Kg={sparseFillEmptyRows:Eg,sparseReshape:Og,sparseSegmentMean:Ag,sparseSegmentSum:Mg},qg={stringNGrams:Pg,stringSplit:Ig,stringToHashBucketFast:Rg,staticRegexReplace:Bg},Jg=new Map,Yg=new Map,Xg=class{getClassName(){return this.constructor.className}static fromConfig(e,t){return new e(t)}},Zg=class e{constructor(){this.classNameMap={}}static getMap(){return e.instance??=new e,e.instance}static register(t){e.getMap().classNameMap[t.className]=[t,t.fromConfig]}};function Qg(e,t,n){y(e.className!=null,()=>`Class being registered does not have the static className property defined.`),y(typeof e.className==`string`,()=>`className is required to be a string, but got type `+typeof e.className),y(e.className.length>0,()=>`Class being registered has an empty-string as its className, which is disallowed.`),t===void 0&&(t=`Custom`),n===void 0&&(n=e.className);let r=n,i=t+`>`+r;return Zg.register(e),Jg.set(i,e),Yg.set(e,i),e}var $g=class extends Xg{minimize(e,t=!1,n){let{value:r,grads:i}=this.computeGradients(e,n);if(n!=null){let e=n.map(e=>({name:e.name,tensor:i[e.name]}));this.applyGradients(e)}else this.applyGradients(i);return F(i),t?r:(r.dispose(),null)}get iterations(){return this.iterations_??=0,this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(e,t){return Mu(e,t)}dispose(){this.iterations_!=null&&F(this.iterations_)}async saveIterations(){return this.iterations_??=0,{name:`iter`,tensor:G(this.iterations_,`int32`)}}async getWeights(){throw Error(`getWeights() is not implemented for this optimizer yet.`)}async setWeights(e){throw Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(e){return this.iterations_=(await e[0].tensor.data())[0],e.slice(1)}};Object.defineProperty($g,Symbol.hasInstance,{value:e=>e.minimize!=null&&e.computeGradients!=null&&e.applyGradients!=null});var e_=class extends $g{static get className(){return`Adadelta`}constructor(e,t,n=null){super(),this.learningRate=e,this.rho=t,this.epsilon=n,this.accumulatedGrads=[],this.accumulatedUpdates=[],n??(this.epsilon=j.backend.epsilon())}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=j.registeredVariables[t];this.accumulatedGrads[n]??(this.accumulatedGrads[n]={originalName:`${t}/accum_grad`,variable:P(()=>ul(r).variable(!1))}),this.accumulatedUpdates[n]??(this.accumulatedUpdates[n]={originalName:`${t}/accum_var`,variable:P(()=>ul(r).variable(!1))});let i=Array.isArray(e)?e[n].tensor:e[t];if(i==null)return;let a=this.accumulatedGrads[n].variable,o=this.accumulatedUpdates[n].variable;P(()=>{let e=R(B(a,this.rho),B(Bl(i),1-this.rho)),t=B(z(Rl(R(o,this.epsilon)),Rl(R(a,this.epsilon))),i),n=R(B(o,this.rho),B(Bl(t),1-this.rho));a.assign(e),o.assign(n);let s=R(B(t,-this.learningRate),r);r.assign(s)})}),this.incrementIterations()}dispose(){this.accumulatedUpdates!=null&&(F(this.accumulatedGrads.map(e=>e.variable)),F(this.accumulatedUpdates.map(e=>e.variable)))}async getWeights(){let e=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);let t=e.length/2;this.accumulatedGrads=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.accumulatedUpdates=e.slice(t,t*2).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.rho,t.epsilon)}},t_=class extends $g{static get className(){return`Adagrad`}constructor(e,t=.1){super(),this.learningRate=e,this.initialAccumulatorValue=t,this.accumulatedGrads=[]}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=j.registeredVariables[t];this.accumulatedGrads[n]??(this.accumulatedGrads[n]={originalName:`${t}/accumulator`,variable:P(()=>mc(r.shape,this.initialAccumulatorValue).variable(!1))});let i=Array.isArray(e)?e[n].tensor:e[t];if(i==null)return;let a=this.accumulatedGrads[n].variable;P(()=>{let e=R(a,Bl(i));a.assign(e);let t=R(B(z(i,Rl(R(e,j.backend.epsilon()))),-this.learningRate),r);r.assign(t)})}),this.incrementIterations()}dispose(){this.accumulatedGrads!=null&&F(this.accumulatedGrads.map(e=>e.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e),this.accumulatedGrads=e.map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(e,t){return new e(t.learningRate,t.initialAccumulatorValue)}},n_=class extends $g{static get className(){return`Adam`}constructor(e,t,n,r=null){super(),this.learningRate=e,this.beta1=t,this.beta2=n,this.epsilon=r,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],P(()=>{this.accBeta1=G(t).variable(),this.accBeta2=G(n).variable()}),r??(this.epsilon=j.backend.epsilon())}applyGradients(e){let t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);P(()=>{let n=q(1,this.accBeta1),r=q(1,this.accBeta2);t.forEach((t,i)=>{let a=j.registeredVariables[t];this.accumulatedFirstMoment[i]??(this.accumulatedFirstMoment[i]={originalName:`${t}/m`,variable:P(()=>ul(a).variable(!1))}),this.accumulatedSecondMoment[i]??(this.accumulatedSecondMoment[i]={originalName:`${t}/v`,variable:P(()=>ul(a).variable(!1))});let o=Array.isArray(e)?e[i].tensor:e[t];if(o==null)return;let s=this.accumulatedFirstMoment[i].variable,c=this.accumulatedSecondMoment[i].variable,l=R(B(s,this.beta1),B(o,1-this.beta1)),u=R(B(c,this.beta2),B(Bl(o),1-this.beta2)),d=z(l,n),f=z(u,r);s.assign(l),c.assign(u);let p=R(B(z(d,R(Rl(f),this.epsilon)),-this.learningRate),a);a.assign(p)}),this.accBeta1.assign(B(this.accBeta1,this.beta1)),this.accBeta2.assign(B(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),this.accumulatedFirstMoment!=null&&F(this.accumulatedFirstMoment.map(e=>e.variable)),this.accumulatedSecondMoment!=null&&F(this.accumulatedSecondMoment.map(e=>e.variable))}async getWeights(){let e=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e),P(()=>{this.accBeta1.assign(Il(this.beta1,this.iterations_+1)),this.accBeta2.assign(Il(this.beta2,this.iterations_+1))});let t=e.length/2;this.accumulatedFirstMoment=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.accumulatedSecondMoment=e.slice(t,t*2).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon)}},r_=class extends $g{static get className(){return`Adamax`}constructor(e,t,n,r=null,i=0){super(),this.learningRate=e,this.beta1=t,this.beta2=n,this.epsilon=r,this.decay=i,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],P(()=>{this.iteration=G(0).variable(),this.accBeta1=G(t).variable()}),r??(this.epsilon=j.backend.epsilon())}applyGradients(e){let t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);P(()=>{let n=q(1,this.accBeta1),r=z(-this.learningRate,R(B(this.iteration,this.decay),1));t.forEach((t,i)=>{let a=j.registeredVariables[t];this.accumulatedFirstMoment[i]??(this.accumulatedFirstMoment[i]={originalName:`${t}/m`,variable:ul(a).variable(!1)}),this.accumulatedWeightedInfNorm[i]??(this.accumulatedWeightedInfNorm[i]={originalName:`${t}/v`,variable:ul(a).variable(!1)});let o=Array.isArray(e)?e[i].tensor:e[t];if(o==null)return;let s=this.accumulatedFirstMoment[i].variable,c=this.accumulatedWeightedInfNorm[i].variable,l=R(B(s,this.beta1),B(o,1-this.beta1)),u=ud(B(c,this.beta2),Ho(o));s.assign(l),c.assign(u);let d=R(B(z(r,n),z(l,R(u,this.epsilon))),a);a.assign(d)}),this.iteration.assign(R(this.iteration,1)),this.accBeta1.assign(B(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),this.accumulatedFirstMoment!=null&&F(this.accumulatedFirstMoment.map(e=>e.variable)),this.accumulatedWeightedInfNorm!=null&&F(this.accumulatedWeightedInfNorm.map(e=>e.variable))}async getWeights(){throw Error(`getWeights() is not implemented for Adamax yet.`)}async setWeights(e){throw Error(`setWeights() is not implemented for Adamax yet.`)}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon,t.decay)}},i_=class extends $g{static get className(){return`SGD`}constructor(e){super(),this.learningRate=e,this.setLearningRate(e)}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=Array.isArray(e)?e[n].tensor:e[t];if(r==null)return;let i=j.registeredVariables[t];P(()=>{let e=R(B(this.c,r),i);i.assign(e)})}),this.incrementIterations()}setLearningRate(e){this.learningRate=e,this.c!=null&&this.c.dispose(),this.c=pa(G(-e))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(e){if(e=await this.extractIterations(e),e.length!==0)throw Error(`SGD optimizer does not have settable weights.`)}getConfig(){return{learningRate:this.learningRate}}static fromConfig(e,t){return new e(t.learningRate)}},a_=[e_,t_,n_,r_,class extends i_{static get className(){return`Momentum`}constructor(e,t,n=!1){super(e),this.learningRate=e,this.momentum=t,this.useNesterov=n,this.accumulations=[],this.m=G(this.momentum)}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=j.registeredVariables[t];this.accumulations[n]??(this.accumulations[n]={originalName:`${t}/momentum`,variable:P(()=>ul(r).variable(!1))});let i=this.accumulations[n].variable,a=Array.isArray(e)?e[n].tensor:e[t];a!=null&&P(()=>{let e,t=R(B(this.m,i),a);e=this.useNesterov?R(B(this.c,R(a,B(t,this.m))),r):R(B(this.c,t),r),i.assign(t),r.assign(e)})}),this.incrementIterations()}dispose(){this.m.dispose(),this.accumulations!=null&&F(this.accumulations.map(e=>e.variable))}setMomentum(e){this.momentum=e}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e),this.accumulations=e.map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(e,t){return new e(t.learningRate,t.momentum,t.useNesterov)}},class extends $g{static get className(){return`RMSProp`}constructor(e,t=.9,n=0,r=null,i=!1){if(super(),this.learningRate=e,this.decay=t,this.momentum=n,this.epsilon=r,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=i,r??(this.epsilon=j.backend.epsilon()),e==null)throw Error(`learningRate for RMSPropOptimizer must be defined.`)}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=j.registeredVariables[t];this.accumulatedMeanSquares[n]??(this.accumulatedMeanSquares[n]={originalName:`${t}/rms`,variable:P(()=>ul(r).variable(!1))}),this.accumulatedMoments[n]??(this.accumulatedMoments[n]={originalName:`${t}/momentum`,variable:P(()=>ul(r).variable(!1))}),this.accumulatedMeanGrads[n]==null&&this.centered&&(this.accumulatedMeanGrads[n]={originalName:`${t}/mg`,variable:P(()=>ul(r).variable(!1))});let i=Array.isArray(e)?e[n].tensor:e[t];if(i==null)return;let a=this.accumulatedMeanSquares[n].variable,o=this.accumulatedMoments[n].variable;P(()=>{let e=R(B(a,this.decay),B(Bl(i),1-this.decay));if(this.centered){let t=this.accumulatedMeanGrads[n].variable,s=R(B(t,this.decay),B(i,1-this.decay)),c=z(B(i,this.learningRate),Rl(q(e,R(Bl(s),this.epsilon)))),l=R(B(o,this.momentum),c);a.assign(e),t.assign(s),o.assign(l);let u=q(r,l);r.assign(u)}else{let e=R(B(a,this.decay),B(Bl(i),1-this.decay)),t=R(B(o,this.momentum),z(B(i,this.learningRate),Rl(R(e,this.epsilon))));a.assign(e),o.assign(t);let n=q(r,t);r.assign(n)}})}),this.incrementIterations()}dispose(){this.accumulatedMeanSquares!=null&&F(this.accumulatedMeanSquares.map(e=>e.variable)),this.accumulatedMeanGrads!=null&&this.centered&&F(this.accumulatedMeanGrads.map(e=>e.variable)),this.accumulatedMoments!=null&&F(this.accumulatedMoments.map(e=>e.variable))}async getWeights(){let e=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&e.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);let t=this.centered?e.length/3:e.length/2;this.accumulatedMeanSquares=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.accumulatedMoments=e.slice(t,t*2).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.centered&&(this.accumulatedMeanGrads=e.slice(t*2,t*3).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(e,t){return new e(t.learningRate,t.decay,t.momentum,t.epsilon,t.centered)}},i_];function o_(){for(let e of a_)Qg(e)}var s_=`model`,c_=`.json`,l_=`.weights.bin`;function u_(e){return new Promise(e=>setTimeout(e)).then(e)}var d_=class e{constructor(t){if(!O().getBool(`IS_BROWSER`))throw Error(`browserDownloads() cannot proceed because the current environment is not a browser.`);t.startsWith(e.URL_SCHEME)&&(t=t.slice(e.URL_SCHEME.length)),(t==null||t.length===0)&&(t=s_),this.modelJsonFileName=t+c_,this.weightDataFileName=t+l_}async save(e){if(typeof document>`u`)throw Error("Browser downloads are not supported in this environment since `document` is not present");let t=ua.join(e.weightData),n=window.URL.createObjectURL(new Blob([t],{type:`application/octet-stream`}));if(e.modelTopology instanceof ArrayBuffer)throw Error(`BrowserDownloads.save() does not support saving model topology in binary formats yet.`);{let t=Ma(e,[{paths:[`./`+this.weightDataFileName],weights:e.weightSpecs}]),r=window.URL.createObjectURL(new Blob([JSON.stringify(t)],{type:`application/json`})),i=this.modelJsonAnchor==null?document.createElement(`a`):this.modelJsonAnchor;if(i.download=this.modelJsonFileName,i.href=r,await u_(()=>i.dispatchEvent(new MouseEvent(`click`))),e.weightData!=null){let e=this.weightDataAnchor==null?document.createElement(`a`):this.weightDataAnchor;e.download=this.weightDataFileName,e.href=n,await u_(()=>e.dispatchEvent(new MouseEvent(`click`)))}return{modelArtifactsInfo:Fa(e)}}}};d_.URL_SCHEME=`downloads://`;var f_=class{constructor(e){if(e==null||e.length<1)throw Error(`When calling browserFiles, at least 1 file is required, but received ${e}`);this.jsonFile=e[0],this.weightsFiles=e.slice(1)}async load(){return new Promise((e,t)=>{let n=new FileReader;n.onload=n=>{let r=JSON.parse(n.target.result),i=r.modelTopology;if(i==null){t(Error(`modelTopology field is missing from file ${this.jsonFile.name}`));return}if(r.weightsManifest==null){t(Error(`weightManifest field is missing from file ${this.jsonFile.name}`));return}if(this.weightsFiles.length===0){e({modelTopology:i});return}e(Pa(r,e=>this.loadWeights(e)))},n.onerror=e=>t(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),n.readAsText(this.jsonFile)})}loadWeights(e){let t=[],n=[];for(let r of e)t.push(...r.weights),n.push(...r.paths);let r=this.checkManifestAndWeightFiles(e),i=n.map(e=>this.loadWeightsFile(e,r[e]));return Promise.all(i).then(e=>[t,e])}loadWeightsFile(e,t){return new Promise((n,r)=>{let i=new FileReader;i.onload=e=>{let t=e.target.result;n(t)},i.onerror=t=>r(`Failed to weights data from file of path '${e}'.`),i.readAsArrayBuffer(t)})}checkManifestAndWeightFiles(e){let t=[],n=this.weightsFiles.map(e=>ja(e.name)),r={};for(let i of e)i.paths.forEach(e=>{let i=ja(e);if(t.indexOf(i)!==-1)throw Error(`Duplicate file basename found in weights manifest: '${i}'`);if(t.push(i),n.indexOf(i)===-1)throw Error(`Weight file with basename '${i}' is not provided.`);r[e]=this.weightsFiles[n.indexOf(i)]});if(t.length!==this.weightsFiles.length)throw Error(`Mismatch in the number of files in weights manifest (${t.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return r}};Va.registerSaveRouter(e=>O().getBool(`IS_BROWSER`)&&!Array.isArray(e)&&e.startsWith(d_.URL_SCHEME)?p_(e.slice(d_.URL_SCHEME.length)):null);function p_(e=`model`){return new d_(e)}function m_(e){return new f_(e)}function h_(e,t,n,r){o(e),n??=0,r??=1,s(n,r);let i=0,a=a=>(a.then(a=>(t(n+ ++i/e.length*(r-n)),a)),a);function o(e){y(e!=null&&Array.isArray(e)&&e.length>0,()=>`promises must be a none empty array`)}function s(e,t){y(e>=0&&e<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${e}`),y(t>=0&&t<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${t}`),y(t>=e,()=>`startFraction must be no more than endFraction, but got startFraction ${e} and endFraction ${t}`)}return Promise.all(e.map(a))}async function g_(e,t){t??={};let n=t.fetchFunc==null?O().platform.fetch:t.fetchFunc,r=e.map(e=>n(e,t.requestInit,{isBinary:!0})),i=(t.onProgress==null?await Promise.all(r):await h_(r,t.onProgress,0,.5)).map(e=>e.arrayBuffer());return t.onProgress==null?await Promise.all(i):await h_(i,t.onProgress,.5,1)}function __(e,t){var n;let r=t.fetchFunc==null?O().platform.fetch:t.fetchFunc,i=0,a;return(n=t.onProgress)==null||n.call(t,0),new ReadableStream({pull:async n=>{for(var o;i<e.length;){a||=(await r(e[i],t.requestInit,{isBinary:!0})).body.getReader();let{done:s,value:c}=await a.read();if(s){i++,a=void 0,(o=t.onProgress)==null||o.call(t,i/e.length);continue}n.enqueue(c);return}n.close()}})}async function v_(e,t=``,n,r){return y_(e=>g_(e,{requestInit:r}))(e,t,n)}function y_(e){return async(t,n=``,r)=>{let i=t.map(()=>!1),a={},o=r==null?[]:r.map(()=>!1),s=[];if(t.forEach((e,t)=>{let n=0;e.weights.forEach(e=>{let c=la[`quantization`in e?e.quantization.dtype:e.dtype]*S(e.shape),l=()=>{i[t]=!0,a[t]??(a[t]=[]),a[t].push({manifestEntry:e,groupOffset:n,sizeBytes:c})};r==null?l():r.forEach((t,n)=>{t===e.name&&(l(),o[n]=!0)}),s.push(e.name),n+=c})}),!o.every(e=>e)){let e=r.filter((e,t)=>!o[t]);throw Error(`Could not find weights in manifest with names: ${e.join(`, `)}. \nManifest JSON has weights with names: ${s.join(`, `)}.`)}let c=i.reduce((e,t,n)=>(t&&e.push(n),e),[]),l=[];c.forEach(e=>{t[e].paths.forEach(e=>{let t=n+(n.endsWith(`/`)?``:`/`)+e;l.push(t)})});let u=await e(l),d={},f=0;return c.forEach(e=>{let n=t[e].paths.length,r=new ua(u.slice(f,f+n));a[e].forEach(e=>{let t=ya(r.slice(e.groupOffset,e.groupOffset+e.sizeBytes),[e.manifestEntry]);for(let e in t)d[e]=t[e]}),f+=n}),d}}var b_=`application/octet-stream`,x_=`application/json`,S_=class{constructor(e,t){if(this.DEFAULT_METHOD=`POST`,t??={},this.weightPathPrefix=t.weightPathPrefix,this.weightUrlConverter=t.weightUrlConverter,t.fetchFunc==null?this.fetch=O().platform.fetch:(y(typeof t.fetchFunc==`function`,()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=t.fetchFunc),y(e!=null&&e.length>0,()=>`URL path for http must not be null, undefined or empty.`),Array.isArray(e)&&y(e.length===2,()=>`URL paths for http must have a length of 2, (actual length is ${e.length}).`),this.path=e,t.requestInit!=null&&t.requestInit.body!=null)throw Error(`requestInit is expected to have no pre-existing body, but has one.`);this.requestInit=t.requestInit||{},this.loadOptions=t}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw Error(`BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.`);let t=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);t.body=new FormData;let n=Ma(e,[{paths:[`./model.weights.bin`],weights:e.weightSpecs}]);if(t.body.append(`model.json`,new Blob([JSON.stringify(n)],{type:x_}),`model.json`),e.weightData!=null){let n=ua.join(e.weightData);t.body.append(`model.weights.bin`,new Blob([n],{type:b_}),`model.weights.bin`)}let r=await this.fetch(this.path,t);if(r.ok)return{modelArtifactsInfo:Fa(e),responses:[r]};throw Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${r.status}.`)}async loadModelJSON(){let e=await this.fetch(this.path,this.requestInit);if(!e.ok)throw Error(`Request to ${this.path} failed with status code ${e.status}. Please verify this URL points to the model JSON of the model to load.`);let t;try{t=await e.json()}catch{let e=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(`.pb`)?e+=` Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.`:e+=` Please make sure the server is serving valid JSON for this request.`,Error(e)}let n=t.modelTopology,r=t.weightsManifest;if(n==null&&r==null)throw Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return t}async load(){return this.loadOptions.streamWeights?this.loadStream():Pa(await this.loadModelJSON(),e=>this.loadWeights(e))}async loadStream(){let e=await this.loadModelJSON(),t=await this.getWeightUrls(e.weightsManifest),n=Ia(e.weightsManifest);return Object.assign(Object.assign({},e),{weightSpecs:n,getWeightStream:()=>__(t,this.loadOptions)})}async getWeightUrls(e){let[t,n]=C_(Array.isArray(this.path)?this.path[1]:this.path),r=this.weightPathPrefix||t,i=[],a=[];for(let t of e)for(let e of t.paths)this.weightUrlConverter==null?i.push(r+e+n):a.push(this.weightUrlConverter(e));return this.weightUrlConverter&&i.push(...await Promise.all(a)),i}async loadWeights(e){let t=await this.getWeightUrls(e);return[Ia(e),await g_(t,this.loadOptions)]}};S_.URL_SCHEME_REGEX=/^https?:\/\//;function C_(e){let t=e.lastIndexOf(`/`),n=e.lastIndexOf(`?`),r=e.substring(0,t),i=n>t?e.substring(n):``;return[r+`/`,i]}function w_(e){return e.match(S_.URL_SCHEME_REGEX)!=null}var T_=(e,t)=>{if(typeof fetch>`u`&&(t==null||t.fetchFunc==null))return null;{let n=!0;if(n=Array.isArray(e)?e.every(e=>w_(e)):w_(e),n)return E_(e,t)}return null};Va.registerSaveRouter(T_),Va.registerLoadRouter(T_);function E_(e,t){return new S_(e,t)}function D_(e,t){return E_(e,t)}var O_=class{constructor(e){this.modelArtifacts=e}load(){return this.modelArtifacts}},k_=class{constructor(e){this.saveHandler=e}save(e){return this.saveHandler(e)}},A_=class{constructor(e){e.load&&(this.load=()=>Promise.resolve(e.load())),e.save&&(this.save=t=>Promise.resolve(e.save(t)))}};function j_(e,t,n,r){return new A_(M_(...arguments))}function M_(e,t,n,r){return arguments.length===1?e.modelTopology!=null||e.weightSpecs!=null?new O_(e):(console.warn(`Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release.`),new O_({modelTopology:e})):(console.warn(`Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release.`),new O_({modelTopology:e,weightSpecs:t,weightData:n,trainingConfig:r}))}function N_(e){return new k_(e)}function P_(e){return new k_(e)}var F_=s({CompositeArrayBuffer:()=>ua,browserFiles:()=>m_,browserHTTPRequest:()=>D_,concatenateArrayBuffers:()=>Aa,copyModel:()=>To,decodeWeights:()=>ya,decodeWeightsStream:()=>wa,encodeWeights:()=>va,fromMemory:()=>j_,fromMemorySync:()=>M_,getLoadHandlers:()=>Ga,getModelArtifactsForJSON:()=>Pa,getModelArtifactsForJSONSync:()=>Na,getModelArtifactsInfoForJSON:()=>Fa,getSaveHandlers:()=>Wa,getWeightSpecs:()=>Ia,http:()=>E_,isHTTPScheme:()=>w_,listModels:()=>Co,loadWeights:()=>v_,moveModel:()=>Eo,registerLoadRouter:()=>Ua,registerSaveRouter:()=>Ha,removeModel:()=>wo,weightsLoaderFactory:()=>y_,withSaveHandler:()=>N_,withSaveHandlerSync:()=>P_}),I_,L_=!1;function R_(e,t=3){if(t>4)throw Error(`Cannot construct Tensor with more than 4 channels from pixels.`);if(e==null)throw Error(`pixels passed to tf.browser.fromPixels() can not be null`);let n=!1,r=!1,i=!1,a=!1,o=!1,s=!1;if(e.data instanceof Uint8Array)n=!0;else if(typeof ImageData<`u`&&e instanceof ImageData)r=!0;else if(typeof HTMLVideoElement<`u`&&e instanceof HTMLVideoElement)i=!0;else if(typeof HTMLImageElement<`u`&&e instanceof HTMLImageElement)a=!0;else if(e.getContext!=null)o=!0;else if(typeof ImageBitmap<`u`&&e instanceof ImageBitmap)s=!0;else throw Error(`pixels passed to tf.browser.fromPixels() must be either an HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData in browser, or OffscreenCanvas, ImageData in webworker or {data: Uint32Array, width: number, height: number}, but was ${e.constructor.name}`);if(Ar(`FromPixels`,j.backendName)!=null){let n={pixels:e},r={numChannels:t};return j.runKernel(xr,n,r)}let[c,l]=i?[e.videoWidth,e.videoHeight]:[e.width,e.height],u;if(o)u=e.getContext(`2d`).getImageData(0,0,c,l).data;else if(r||n)u=e.data;else if(a||i||s){if(I_==null)if(typeof document>`u`)if(typeof OffscreenCanvas<`u`&&typeof OffscreenCanvasRenderingContext2D<`u`)I_=new OffscreenCanvas(1,1).getContext(`2d`);else throw Error(`Cannot parse input in current context. Reason: OffscreenCanvas Context2D rendering is not supported.`);else I_=document.createElement(`canvas`).getContext(`2d`,{willReadFrequently:!0});I_.canvas.width=c,I_.canvas.height=l,I_.drawImage(e,0,0,c,l),u=I_.getImageData(0,0,c,l).data}let d;if(t===4)d=new Int32Array(u);else{let e=c*l;d=new Int32Array(e*t);for(let n=0;n<e;n++)for(let e=0;e<t;++e)d[n*t+e]=u[n*4+e]}return qp(d,[l,c,t],`int32`)}function z_(e){if(e.rank!==2&&e.rank!==3)throw Error(`toPixels only supports rank 2 or 3 tensors, got rank ${e.rank}.`);let t=e.rank===2?1:e.shape[2];if(t>4||t===2)throw Error(`toPixels only supports depth of size 1, 3 or 4 but got ${t}`);if(e.dtype!==`float32`&&e.dtype!==`int32`)throw Error(`Unsupported type for toPixels: ${e.dtype}. Please use float32 or int32 tensors.`)}async function B_(e,t){let n=M(e,`img`,`toPixels`);if(!(e instanceof Ei)){let e=n;n=L(e,`int32`),e.dispose()}z_(n);let[r,i]=n.shape.slice(0,2),a=n.rank===2?1:n.shape[2],o=await n.data(),s=n.dtype===`float32`?255:1,c=new Uint8ClampedArray(i*r*4);for(let e=0;e<r*i;++e){let t=[0,0,0,255];for(let r=0;r<a;r++){let i=o[e*a+r];if(n.dtype===`float32`){if(i<0||i>1)throw Error(`Tensor values for a float32 Tensor must be in the range [0 - 1] but encountered ${i}.`)}else if(n.dtype===`int32`&&(i<0||i>255))throw Error(`Tensor values for a int32 Tensor must be in the range [0 - 255] but encountered ${i}.`);a===1?(t[0]=i*s,t[1]=i*s,t[2]=i*s):t[r]=i*s}let r=e*4;c[r+0]=Math.round(t[0]),c[r+1]=Math.round(t[1]),c[r+2]=Math.round(t[2]),c[r+3]=Math.round(t[3])}if(t!=null){L_||Ar(`Draw`,j.backendName)!=null&&(console.warn(`tf.browser.toPixels is not efficient to draw tensor on canvas. Please try tf.browser.draw instead.`),L_=!0),t.width=i,t.height=r;let e=t.getContext(`2d`),n=new ImageData(c,i,r);e.putImageData(n,0,0)}return n!==e&&n.dispose(),c}var V_=N({fromPixels_:R_});function H_(e,t){let n=e.shape.length,r=t.shape.length;if(n<1)throw Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${n}.`);if(r<1)throw Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${r}.`);if(t.dtype!==`int32`)throw Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${t.dtype}.`);if(t.shape[r-1]>n)throw Error(`index innermost dimension length must be <= tensor rank; saw: ${t.shape[r-1]} vs. ${n}`);if(S(e.shape)===0)throw Error(`Requested more than 0 entries, but input is empty. Input shape: ${e.shape}.`);let i=t.shape,a=i[i.length-1],o=1;for(let e=0;e<i.length-1;++e)o*=i[e];let s=e.shape,c=i.slice();c.pop();let l=1;for(let e=a;e<n;++e)l*=s[e],c.push(s[e]);let u=[..._e(e.shape).map(e=>e/l),1].slice(0,a);return[c,o,l,u]}var U_=s({assertParamsValid:()=>K_,computeFlatOffset:()=>av,computeOutShape:()=>J_,getNormalizedAxes:()=>Q_,isSliceContinous:()=>iv,maskToAxes:()=>q_,parseSliceParams:()=>ov,sliceInfo:()=>sv,startForAxis:()=>nv,startIndicesWithElidedDims:()=>$_,stopForAxis:()=>rv,stopIndicesWithElidedDims:()=>ev,stridesForAxis:()=>tv,stridesWithElidedDims:()=>Y_}),W_=-2,G_=-1;function K_(e,t,n){let r=e.shape.length;y(r===t.length,()=>`Error in slice${r}D: Length of begin ${t} must match the rank of the array (${r}).`),y(r===n.length,()=>`Error in slice${r}D: Length of size ${n} must match the rank of the array (${r}).`);for(let i=0;i<r;++i)y(t[i]+n[i]<=e.shape[i],()=>`Error in slice${r}D: begin[${i}] + size[${i}] (${t[i]+n[i]}) would overflow input.shape[${i}] (${e.shape[i]})`)}function q_(e){let t=[],n=0;for(;e>0;)e&1&&t.push(n),e/=2,n++;return t}function J_(e,t,n){let r=[];for(let i=0;i<e.length;i++)r[i]=Math.ceil((t[i]-e[i])/n[i]);return r}function Y_(e,t,n,r){let i=[...e];for(let e=i.length;e<r.length;e++)i.push(1);for(let e=0;e<n;e++)e===0?i[t]=1:(i.splice(t,0,1),i.pop());return i}function X_(e,t,n){return n<=e?n:n-(t-1)}function Z_(e,t){let n=[];for(let r=0;r<e;r++)n.push(t+r);return n}function Q_(e,t,n,r,i,a,o,s,c){let l=e.length,u=Array(l),d=Array(l),f=Array(l);if(t.length&&n>0){let c=t[0],l=n+1;u=$_(o,c,l,r,e),d=ev(s,c,l,i,e),f=Y_(a,c,l,e)}else for(let t=0;t<l;t++)u[t]=nv(o,r,a,e,t,c),d[t]=rv(s,i,a,e,t,c),f[t]=tv(a,t,c);return{begin:u,end:d,strides:f}}function $_(e,t,n,r,i){let a=[...i],o=Z_(n,t);for(let i=0;i<a.length;i++)if(o.indexOf(i)>-1)a[i]=0;else{let o=X_(t,n,i),s=r[o];e&1<<o&&(s=0),a[i]=s}return a}function ev(e,t,n,r,i){let a=[...i],o=Z_(n,t);for(let i=0;i<a.length;i++)if(o.indexOf(i)>-1)a[i]=2**53-1;else{let o=X_(t,n,i),s=r[o];e&1<<o&&(s=2**53-1),a[i]=s}for(let e=0;e<a.length;e++){let t=i[e];a[e]<0&&(a[e]+=t),a[e]=h(0,a[e],i[e])}return a}function tv(e,t,n){let r=e[t];return(n&1<<t||r==null)&&(r=1),r}function nv(e,t,n,r,i,a){let o=t[i],s=n[i]||1;(e&1<<i||a&1<<i||o==null)&&(o=s>0?-(2**53-1):2**53-1);let c=r[i];return o<0&&(o+=c),o=h(0,o,c-1),o}function rv(e,t,n,r,i,a){let o=t[i],s=n[i]||1;(e&1<<i||a&1<<i||o==null)&&(o=s>0?2**53-1:-(2**53-1));let c=r[i];return o<0&&(o+=c),o=s>0?h(0,o,c):h(-1,o,c-1),o}function iv(e,t,n){let r=n.length;for(let e=0;e<n.length;e++)if(n[e]>1){r=e;break}for(let i=r+1;i<n.length;i++)if(t[i]>0||n[i]!==e[i])return!1;return!0}function av(e,t){let n=e.length>0?e[e.length-1]:1;for(let r=0;r<e.length-1;r++)n+=e[r]*t[r];return n}function ov(e,t,n){let r,i=e.shape.length;r=typeof t==`number`?[t,...Array(i-1).fill(0)]:t.length<i?t.concat(Array(i-t.length).fill(0)):t.slice(),r.forEach(e=>{y(e!==-1,()=>`slice() does not support negative begin indexing.`)});let a;return a=n==null?Array(i).fill(-1):typeof n==`number`?[n,...Array(i-1).fill(-1)]:n.length<i?n.concat(Array(i-n.length).fill(-1)):n,a=a.map((t,n)=>t>=0?t:(y(t===-1,()=>`Negative size values should be exactly -1 but got ${t} for the slice() size at index ${n}.`),e.shape[n]-r[n])),[r,a]}function sv(e,t,n,r,i,a,o,s,c){let l;if(r==null?(l=Array(t.length),l.fill(1)):l=r,o!=null&&o&o-1)throw Error(`Multiple ellipses in slice is not allowed.`);let u=!1,d={dims:l.length,numAddAxisAfterEllipsis:0,begin:t.slice(),end:n.slice(),strides:l.slice(),beginMask:i,endMask:a,ellipsisMask:o,newAxisMask:s,shrinkAxisMask:c};for(let e=0;e<d.dims;e++)u&&1<<e&s&&d.numAddAxisAfterEllipsis++,1<<e&o&&(u=!0);u||(d.ellipsisMask|=1<<d.dims,d.dims++);let f={dims:e.length,beginMask:0,endMask:0,beginValid:!1,endValid:!1};cv(d,f);let p=!0,m=!0,h=!0,g=[],_=[];for(let t=0;t<e.length;++t){if(f.strides[t]===0)throw Error(`strides[${t}] must be non-zero`);let n=!!(f.shrinkAxisMask&1<<t),r=e[t];if(r===-1){g.push(n?1:-1);continue}let i=[f.beginMask&1<<t,f.endMask&1<<t],a=[f.strides[t]>0?0:-1,f.strides[t]>0?r:r-1];if(n&&f.strides[t]<=0)throw Error(`only stride 1 allowed on non-range indexing.`);h&&=f.strides[t]===1;let o=!!(f.beginMask&1<<t&&f.endMask&1<<t);if(f.beginValid&&f.endValid){if(n){let e=f.begin[t]<0?r+f.begin[t]:f.begin[t];if(f.begin[t]=e,f.end[t]=f.begin[t]+1,e<0||e>=r)throw Error(`slice index ${f.begin[t]} of dimension ${t} out of bounds.`)}else f.begin[t]=lv(f.begin[t],0,f.strides[t],r,i,a),f.end[t]=lv(f.end[t],1,f.strides[t],r,i,a);let e=f.strides[t]===1&&f.begin[t]===0&&f.end[t]===r;p&&=e,m&&=t===0&&f.strides[t]===1||e}else p=p&&f.strides[t]===1&&o,m&&=t===0&&f.strides[t]===1||o;let s,c=!1;if(f.beginValid&&f.endValid?(s=f.end[t]-f.begin[t],c=!0):n?(s=1,c=!0):o&&r>=0&&(s=f.strides[t]<0?-r:r,c=!0),c){let e;e=s===0||s<0!=f.strides[t]<0?0:Math.trunc(s/f.strides[t])+(s%f.strides[t]===0?0:1),g.push(e)}else g.push(-1)}for(let e=0;e<f.finalShapeGatherIndices.length;++e){let t=f.finalShapeGatherIndices[e];t>=0?_.push(g[t]):t===W_&&_.push(1)}return{finalShapeSparse:_.filter((e,t)=>f.finalShapeGatherIndices[t]!==W_),finalShape:_,isIdentity:p,sliceDim0:m,isSimpleSlice:h,begin:f.begin,end:f.end,strides:f.strides}}function cv(e,t){t.beginMask=0,t.endMask=0,t.shrinkAxisMask=0;let n=0;t.beginValid=e.begin!=null,t.endValid=e.end!=null,t.begin=Array(t.dims),t.end=Array(t.dims),t.strides=Array(t.dims),t.finalShapeGatherIndices=[],t.finalShapeGatherIndicesSparse=[],t.inputShapeGatherIndicesSparse=Array(t.dims);for(let r=0;r<e.dims;r++)if(1<<r&e.ellipsisMask){let i=Math.min(t.dims-(e.dims-r)+1+e.numAddAxisAfterEllipsis,t.dims);for(;n<i;n++)t.begin[n]=0,t.end[n]=0,t.strides[n]=1,t.beginMask|=1<<n,t.endMask|=1<<n,t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(-1),t.inputShapeGatherIndicesSparse[n]=r}else if(1<<r&e.newAxisMask)t.finalShapeGatherIndices.push(W_),t.finalShapeGatherIndicesSparse.push(-1);else{if(n===t.begin.length)throw Error(`Index out of range using input dim ${n}; input has only ${t.dims} dims, ${t.begin.length}.`);e.begin!=null&&(t.begin[n]=e.begin[r]),e.end!=null&&(t.end[n]=e.end[r]),t.strides[n]=e.strides[r],e.beginMask&1<<r&&(t.beginMask|=1<<n),e.endMask&1<<r&&(t.endMask|=1<<n),e.shrinkAxisMask&1<<r?(t.finalShapeGatherIndices.push(G_),t.finalShapeGatherIndicesSparse.push(-1),t.shrinkAxisMask|=1<<n):(t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(r)),t.inputShapeGatherIndicesSparse[n]=r,n++}}function lv(e,t,n,r,i,a){if(i[t])return n>0?a[t]:a[t+1&1];{let t=e<0?r+e:e;return t<a[0]?a[0]:t>a[1]?a[1]:t}}var uv=typeof requestAnimationFrame<`u`?requestAnimationFrame:typeof setImmediate<`u`?setImmediate:e=>e();function dv(){return new Promise(e=>uv(()=>e()))}function fv(e,t){let n=e[0].length;e.forEach((e,t)=>{y(e.length===n,()=>`Error in concat${n}D: rank of tensors[${t}] must be the same as the rank of the rest (${n})`)}),y(t>=0&&t<n,()=>`Error in concat${n}D: axis must be between 0 and ${n-1}.`);let r=e[0];e.forEach((e,i)=>{for(let a=0;a<n;a++)y(a===t||e[a]===r[a],()=>`Error in concat${n}D: Shape of tensors[${i}] (${e}) does not match the shape of the rest (${r}) along the non-concatenated axis ${i}.`)})}function pv(e,t){let n=e[0].slice();for(let r=1;r<e.length;r++)n[t]+=e[r][t];return n}var mv;(function(e){e[e.FIRST_DIM_SIZE=0]=`FIRST_DIM_SIZE`,e[e.VALUE_ROWIDS=1]=`VALUE_ROWIDS`,e[e.ROW_LENGTHS=2]=`ROW_LENGTHS`,e[e.ROW_SPLITS=3]=`ROW_SPLITS`,e[e.ROW_LIMITS=4]=`ROW_LIMITS`,e[e.ROW_STARTS=5]=`ROW_STARTS`})(mv||={});function hv(e,t,n){let r=[];if(n==null&&t==null)return r;if(t==null)for(;r.length<e+n.length;)r.push(-1);else r=t.slice();if(n==null)return r;if(e+n.length!==r.length)throw Error(`rt input.shape and shape=${t} are incompatible: rt input.rank = ${e+n.length}, but shape.rank = ${r.length}`);for(let i=1;i<n.length;++i){let a=n[i],o=r[r.length-n.length+i],s=r[o];if(a>=0)if(s>=0){if(s!==a)throw Error(`rt input.shape and shape=${t} are incompatible: rt input.shape[${i+e}] = ${a} but shape[${i+e}] = ${s}`)}else r[o]=a}return r}function gv(e){let t={FIRST_DIM_SIZE:mv.FIRST_DIM_SIZE,VALUE_ROWIDS:mv.VALUE_ROWIDS,ROW_LENGTHS:mv.ROW_LENGTHS,ROW_SPLITS:mv.ROW_SPLITS,ROW_LIMITS:mv.ROW_LIMITS,ROW_STARTS:mv.ROW_STARTS},n=[];for(let r of e)if(r in t)n.push(t[r]);else break;return n}function _v(e){return e.length===0?0:e[0]===mv.FIRST_DIM_SIZE?e.length-1:e.length}function vv(e,t){if(e==null||t==null)return;let n=e.length,r=t.length;if(n>=r)throw Error(`defaultValue.shape=${e} and ragged tensor flatValues.shape=${t}, are incompatible: defaultValue.rank = ${n} must be less than ragged tensor input flatValues.rank = ${r})`);for(let i=0;i<Math.min(n,r-1);++i){let n=e[i],r=t[i+1];if(n>=0&&r>=0&&n!==1&&n!==r)throw Error(`defaultValue.shape=${e}, and ragged tensor input flatValues.shape=${t} are incompatible: defaultValue.shape[${i-e.length}] = ${n} but ragged tensor input.flatValues.shape[${i-e.length}] = ${r}`)}}function yv(e){return e<=30?e:ge(e,Math.floor(Math.sqrt(e)))}function bv(e,t,n){return[n*(typeof e==`number`?e:e[0]),t*(typeof e==`number`?e:e[1])]}function xv(e,t,n,r=!0){let i=[];if(r)i=i.concat(t.slice(0)),i.push(e[0]/n),i=i.concat(e.slice(1));else{i=i.concat(e[0]);let n=t.length;for(let r=0;r<n;++r)i=i.concat([e[r+1]/t[r],t[r]]);i=i.concat(e.slice(n+1))}return i}function Sv(e,t,n=!0){let r=[];if(n){r.push(t);for(let n=t+1;n<e;++n)n<=2*t?(r.push(n),r.push(n-(t+1))):r.push(n)}else{let n=[],i=[];for(let r=1;r<e;++r)r>=t*2+1||r%2==1?i.push(r):n.push(r);r.push(...n),r.push(0),r.push(...i)}return r}function Cv(e,t,n,r=!0){let i=[];r?i.push(e[0]/n):i.push(e[0]*n);for(let n=1;n<e.length;++n)n<=t.length?r?i.push(t[n-1]*e[n]):i.push(e[n]/t[n-1]):i.push(e[n]);return i}function wv(e,t){let n=[0];for(let r=0;r<t;++r)n.push(e[r][0]);return n}function Tv(e,t,n){let r=e.slice(0,1);for(let i=0;i<n;++i)r.push(e[i+1]-t[i][0]-t[i][1]);return r}var Ev=1.7580993408473768,Dv=1.0507009873554805,Ov=.3275911,kv=.254829592,Av=-.284496736,jv=1.421413741,Mv=-1.453152027,Nv=1.061405429;function Pv(e,t){if(e.length!==t.length)throw Error(`Cannot merge real and imag arrays of different lengths. real:${e.length}, imag: ${t.length}.`);let n=new Float32Array(e.length*2);for(let r=0;r<n.length;r+=2)n[r]=e[r/2],n[r+1]=t[r/2];return n}function Fv(e){let t=new Float32Array(e.length/2),n=new Float32Array(e.length/2);for(let r=0;r<e.length;r+=2)t[r/2]=e[r],n[r/2]=e[r+1];return{real:t,imag:n}}function Iv(e){let t=Math.ceil(e.length/4),n=new Float32Array(t),r=new Float32Array(t);for(let t=0;t<e.length;t+=4)n[Math.floor(t/4)]=e[t],r[Math.floor(t/4)]=e[t+1];return{real:n,imag:r}}function Lv(e){let t=Math.floor(e.length/4),n=new Float32Array(t),r=new Float32Array(t);for(let t=2;t<e.length;t+=4)n[Math.floor(t/4)]=e[t],r[Math.floor(t/4)]=e[t+1];return{real:n,imag:r}}function Rv(e,t){return{real:e[t*2],imag:e[t*2+1]}}function zv(e,t,n,r){e[r*2]=t,e[r*2+1]=n}function Bv(e,t){let n=new Float32Array(e/2),r=new Float32Array(e/2);for(let i=0;i<Math.ceil(e/2);i++){let a=(t?2:-2)*Math.PI*(i/e);n[i]=Math.cos(a),r[i]=Math.sin(a)}return{real:n,imag:r}}function Vv(e,t,n){let r=(n?2:-2)*Math.PI*(e/t);return{real:Math.cos(r),imag:Math.sin(r)}}var Hv=`->`,Uv=/->/g,Wv=`,`,Gv=`...`;function Kv(e,t){e=e.replace(/\s/g,``);let n=(e.length-e.replace(Uv,``).length)/2;if(n<1)throw Error(`Equations without an arrow are not supported.`);if(n>1)throw Error(`Equation must contain exactly one arrow ("${Hv}").`);let[r,i]=e.split(Hv);y(r.indexOf(Gv)===-1,()=>`The ellipsis notation ("${Gv}") is not supported yet.`);let a=r.split(Wv),o=a.length;if(t!==o)throw Error(`Expected ${o} input tensors, received ${t}`);if(o>2)throw Error(`Support for more than 2 input tensors is not implemented yet.`);let s=[];for(let e=0;e<i.length;++e){let t=i[e];if(!a.some(e=>e.indexOf(t)!==-1))throw Error(`Output subscripts contain the label ${t} not present in the input subscripts.`);s.indexOf(t)===-1&&s.push(t)}for(let e=0;e<r.length;++e){let t=r[e];s.indexOf(t)===-1&&t!==Wv&&s.push(t)}let c=Array(a.length);for(let e=0;e<o;++e){if(new Set(a[e].split(``)).size!==a[e].length)throw Error(`Found duplicate axes in input component ${a[e]}. Support for duplicate axes in input is not implemented yet.`);c[e]=[];for(let t=0;t<a[e].length;++t)c[e].push(s.indexOf(a[e][t]))}let l=s.length,u=i.length,d=[];for(let e=u;e<l;++e)d.push(e);return{allDims:s,summedDims:d,idDims:c}}function qv(e,t){let n=Array(e);n.fill(-1);for(let e=0;e<t.length;++e)n[t[e]]=e;let r=[];for(let t=0;t<e;++t)n[t]===-1&&r.push(t);return n=n.filter(e=>e!==-1),{permutationIndices:n,expandDims:r}}function Jv(e,t,n){let r=Array(e);for(let e=0;e<n.length;++e){let i=n[e].shape;for(let n=0;n<t[e].length;++n)r[t[e][n]]===void 0?r[t[e][n]]=i[n]:y(r[t[e][n]]===i[n],()=>`Expected dimension ${r[t[e][n]]} at axis ${n} of input shaped ${JSON.stringify(i)}, but got dimension ${i[n]}`)}}function Yv(e,t){let n=e,r=[],i=0;e.length===0&&n.push(-1),i=e.length+1;for(let e=0;e<i;++e)r.push([]);let a=[];for(let e=0;e<n.length;++e){let i=n[e],o=Zv(t,i);for(let t of o)a.indexOf(t)===-1&&(r[e].push(t),a.push(t))}return{path:n,steps:r}}function Xv(e){return e.every((e,t)=>e===t)}function Zv(e,t){let n=[];for(let r=0;r<e.length;++r)(e[r].length===0||e[r].indexOf(t)!==-1||t===-1)&&n.push(r);return n}function Qv(e,t,n=0){let r=[];if(typeof t==`number`)y(e.shape[n]%t===0,()=>`Number of splits must evenly divide the axis.`),r=Array(t).fill(e.shape[n]/t);else{y(t.reduce((e,t)=>(t===-1&&(e+=1),e),0)<=1,()=>`There should be only one negative value in split array.`);let i=t.indexOf(-1);if(i!==-1){let r=t.reduce((e,t)=>t>0?e+t:e);t[i]=e.shape[n]-r}y(e.shape[n]===t.reduce((e,t)=>e+t),()=>`The sum of sizes must match the size of the axis dimension.`),r=t}return r}function $v(e){return`Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${e}`}function ey(e,t){return`indices(${e}, 0) is invalid: ${t} < 0`}function ty(e,t,n){return`indices(${e}, 0) is invalid: ${t} >= ${n}`}function ny(e,t){return`only one output dimension may be -1, not both ${e} and ${t}`}function ry(e,t){return`size ${e} must be non-negative, not ${t}`}function iy(){return`reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero`}function ay(e,t){return`Input to reshape is a SparseTensor with ${S(e)}
  dense values, but the requested shape requires a multiple of ${S(t)}. inputShape=${e} outputShape= ${t}`}function oy(e,t){return`Input to reshape is a tensor with ${S(e)} dense values, but the requested shape has ${S(t)}. inputShape=${e} outputShape=${t}`}function sy(){return`segment ids must be >= 0`}function cy(){return`segment ids are not increasing`}function ly(e,t){return`Segment id ${e} out of range [0, ${t}), possibly because segmentIds input is not sorted.`}function uy(e,t,n){return`Bad: indices[${e}] == ${t} out of range [0, ${n})`}var dy=s({collectGatherOpShapeInfo:()=>my,computeOutShape:()=>py,segOpComputeOptimalWindowSize:()=>fy});function fy(e,t){let n=!1,r;for(e<=30?(r=e,n=!0):r=ge(e,Math.floor(Math.sqrt(e)));!n;)r>t||r===e?n=!0:r=ge(e,r+1);return r}function py(e,t,n){let r=[],i=e.length;for(let a=0;a<i;a++)a===t?r.push(n):r.push(e[a]);return r}function my(e,t,n,r){let i=t.shape.length,a=e.shape.length;if(r!==0&&(r<-i||r>i))throw Error(`Expect batchDims in the range of [-${i}, ${i}], but got ${r}`);if(r<0&&(r+=i),r>a)throw Error(`batchDims (${r}) must be less than rank(x) (
    ${a}).`);if(n<r)throw Error(`batchDims (${r}) must be less than or equal to axis (${n}).`);for(let n=0;n<r;++n)if(e.shape[n]!==t.shape[n])throw Error(`x.shape[${n}]: ${e.shape[n]} should be equal to indices.shape[${n}]: ${t.shape[n]}.`);let o=e.shape[n],s=[],c=1,l=1,u=1;for(let t=0;t<r;++t)s.push(e.shape[t]),c*=e.shape[t];for(let t=r;t<n;t++)s.push(e.shape[t]),l*=e.shape[t];for(let e=r;e<i;e++)s.push(t.shape[e]);for(let t=n+1;t<a;t++)s.push(e.shape[t]),u*=e.shape[t];return{batchSize:c,sliceSize:u,outerSize:l,dimSize:o,outputShape:s}}var hy=s({ERF_A1:()=>kv,ERF_A2:()=>Av,ERF_A3:()=>jv,ERF_A4:()=>Mv,ERF_A5:()=>Nv,ERF_P:()=>Ov,PARALLELIZE_THRESHOLD:()=>30,RowPartitionType:()=>mv,SELU_SCALE:()=>Dv,SELU_SCALEALPHA:()=>Ev,applyActivation:()=>Vm,assertAndGetBroadcastShape:()=>W,assertAxesAreInnerMostDims:()=>Dl,assertParamsConsistent:()=>fv,assignToTypedArray:()=>zv,axesAreInnerMostDims:()=>Cl,calculateShapes:()=>$p,checkEinsumDimSizes:()=>Jv,checkPadOnDimRoundingMode:()=>js,combineLocations:()=>wl,combineRaggedTensorToTensorShapes:()=>hv,complexWithEvenIndex:()=>Iv,complexWithOddIndex:()=>Lv,computeConv2DInfo:()=>gs,computeConv3DInfo:()=>_s,computeDefaultPad:()=>bs,computeDilation2DInfo:()=>ps,computeOptimalWindowSize:()=>yv,computeOutAndReduceShapes:()=>Tl,computeOutShape:()=>pv,computePool2DInfo:()=>ms,computePool3DInfo:()=>hs,convertConv2DDataFormat:()=>As,decodeEinsumEquation:()=>Kv,eitherStridesOrDilationsAreOne:()=>Os,expandShapeToKeepDim:()=>El,exponent:()=>Vv,exponents:()=>Bv,fromStringArrayToUint8:()=>_y,fromUint8ToStringArray:()=>gy,getAxesPermutation:()=>Ol,getBroadcastDims:()=>rl,getComplexWithIndex:()=>Rv,getEinsumComputePath:()=>Yv,getEinsumPermutation:()=>qv,getFusedBiasGradient:()=>Bm,getFusedDyActivation:()=>zm,getImageCenter:()=>bv,getInnerMostAxes:()=>Al,getPermuted:()=>Sv,getRaggedRank:()=>_v,getReductionAxes:()=>il,getReshaped:()=>xv,getReshapedPermuted:()=>Cv,getRowPartitionTypesHelper:()=>gv,getSliceBeginCoords:()=>wv,getSliceSize:()=>Tv,getSparseFillEmptyRowsIndicesDenseShapeMismatch:()=>$v,getSparseFillEmptyRowsNegativeIndexErrorMessage:()=>ey,getSparseFillEmptyRowsOutOfRangeIndexErrorMessage:()=>ty,getSparseReshapeEmptyTensorZeroOutputDimErrorMessage:()=>iy,getSparseReshapeInputOutputMismatchErrorMessage:()=>oy,getSparseReshapeInputOutputMultipleErrorMessage:()=>ay,getSparseReshapeMultipleNegativeOneOutputDimErrorMessage:()=>ny,getSparseReshapeNegativeOutputDimErrorMessage:()=>ry,getSparseSegmentReductionIndicesOutOfRangeErrorMessage:()=>uy,getSparseSegmentReductionNegativeSegmentIdsErrorMessage:()=>sy,getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage:()=>cy,getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage:()=>ly,getUndoAxesPermutation:()=>kl,isIdentityPermutation:()=>Xv,log:()=>Dr,mergeRealAndImagArrays:()=>Pv,prepareAndValidate:()=>H_,prepareSplitSize:()=>Qv,segment_util:()=>dy,shouldFuse:()=>Hm,slice_util:()=>U_,splitRealAndImagArrays:()=>Fv,stridesOrDilationsArePositive:()=>ks,tupleValuesAreOne:()=>Ds,upcastType:()=>Fi,validateDefaultValueShape:()=>vv,validateInput:()=>Qp,validateUpdateShape:()=>Zp,warn:()=>Er});function gy(e){try{return e.map(e=>ii(e))}catch(e){throw Error(`Failed to decode encoded string bytes into utf-8, error: ${e}`)}}function _y(e){return e.map(e=>ri(e))}o_(),O().registerFlag(`KEEP_INTERMEDIATE_TENSORS`,()=>!1,e=>{e&&console.warn(`Keep intermediate tensors is ON. This will print the values of all intermediate tensors during model inference. Not all models support this mode. For details, check e2e/benchmarks/ model_config.js. This significantly impacts performance.`)});var vy;(function(e){e[e.DT_INVALID=0]=`DT_INVALID`,e[e.DT_FLOAT=1]=`DT_FLOAT`,e[e.DT_DOUBLE=2]=`DT_DOUBLE`,e[e.DT_INT32=3]=`DT_INT32`,e[e.DT_UINT8=4]=`DT_UINT8`,e[e.DT_INT16=5]=`DT_INT16`,e[e.DT_INT8=6]=`DT_INT8`,e[e.DT_STRING=7]=`DT_STRING`,e[e.DT_COMPLEX64=8]=`DT_COMPLEX64`,e[e.DT_INT64=9]=`DT_INT64`,e[e.DT_BOOL=10]=`DT_BOOL`,e[e.DT_QINT8=11]=`DT_QINT8`,e[e.DT_QUINT8=12]=`DT_QUINT8`,e[e.DT_QINT32=13]=`DT_QINT32`,e[e.DT_BFLOAT16=14]=`DT_BFLOAT16`,e[e.DT_QINT16=15]=`DT_QINT16`,e[e.DT_QUINT16=16]=`DT_QUINT16`,e[e.DT_UINT16=17]=`DT_UINT16`,e[e.DT_COMPLEX128=18]=`DT_COMPLEX128`,e[e.DT_HALF=19]=`DT_HALF`,e[e.DT_RESOURCE=20]=`DT_RESOURCE`,e[e.DT_VARIANT=21]=`DT_VARIANT`,e[e.DT_UINT32=22]=`DT_UINT32`,e[e.DT_UINT64=23]=`DT_UINT64`,e[e.DT_FLOAT_REF=101]=`DT_FLOAT_REF`,e[e.DT_DOUBLE_REF=102]=`DT_DOUBLE_REF`,e[e.DT_INT32_REF=103]=`DT_INT32_REF`,e[e.DT_UINT8_REF=104]=`DT_UINT8_REF`,e[e.DT_INT16_REF=105]=`DT_INT16_REF`,e[e.DT_INT8_REF=106]=`DT_INT8_REF`,e[e.DT_STRING_REF=107]=`DT_STRING_REF`,e[e.DT_COMPLEX64_REF=108]=`DT_COMPLEX64_REF`,e[e.DT_INT64_REF=109]=`DT_INT64_REF`,e[e.DT_BOOL_REF=110]=`DT_BOOL_REF`,e[e.DT_QINT8_REF=111]=`DT_QINT8_REF`,e[e.DT_QUINT8_REF=112]=`DT_QUINT8_REF`,e[e.DT_QINT32_REF=113]=`DT_QINT32_REF`,e[e.DT_BFLOAT16_REF=114]=`DT_BFLOAT16_REF`,e[e.DT_QINT16_REF=115]=`DT_QINT16_REF`,e[e.DT_QUINT16_REF=116]=`DT_QUINT16_REF`,e[e.DT_UINT16_REF=117]=`DT_UINT16_REF`,e[e.DT_COMPLEX128_REF=118]=`DT_COMPLEX128_REF`,e[e.DT_HALF_REF=119]=`DT_HALF_REF`,e[e.DT_RESOURCE_REF=120]=`DT_RESOURCE_REF`,e[e.DT_VARIANT_REF=121]=`DT_VARIANT_REF`,e[e.DT_UINT32_REF=122]=`DT_UINT32_REF`,e[e.DT_UINT64_REF=123]=`DT_UINT64_REF`})(vy||={});var yy;(function(e){(function(e){e[e.LEGACY=0]=`LEGACY`,e[e.V1=1]=`V1`,e[e.V2=2]=`V2`})(e.CheckpointFormatVersion||={})})(yy||={});var by={};function xy(e){return by[e]}function J(e,t,n,r,i){let a=t.inputParams[e];if(a&&a.inputIndexStart!==void 0){let e=a.inputIndexStart,o=a.inputIndexEnd===0?void 0:a.inputIndexEnd===void 0?e+1:a.inputIndexEnd,s=e<0?t.inputNames.length+e:e;if(a.type===`tensor`)return Sy(t.inputNames[s],n,r,i);if(a.type===`tensors`){let a=t.inputs.slice(e,o);return t.inputNames.slice(e,o).filter((e,t)=>a[t]?.op!==`NoOp`).map(e=>Sy(e,n,r,i))}let c=Sy(t.inputNames[s],n,r,i),l=c.dataSync();return a.type===`number`?l[0]:ye(c.shape,l)}let o=t.attrParams[e];return o&&o.value}function Sy(e,t,n,r){let[i,a]=Ey(e,n);if(r!=null){let e=r.getHashTableHandleByName(i);if(e!=null)return e}let o=n.currentContextIds.find(e=>!!t[Ty(i,e)]);return o===void 0?void 0:t[Ty(i,o)][a]}function Cy(e,t,n){return t[Ty(e,n.currentContextId)]}function wy(e,t){let[n,r,i]=Ey(e,t);return[Ty(n,t&&t.currentContextId),r,i]}function Ty(e,t){return t?`${e}-${t}`:e}function Ey(e,t){if(e===``)return[``,0,void 0];let n=t!=null&&t.parseNodeNameCache!=null;if(n){let n=t.parseNodeNameCache.get(e);if(n!=null)return n}let r=e.split(`:`),i;if(r.length===1)i=[e,0,void 0];else{let e=r[0],t=r.length===3?r[1]:void 0;i=[e,Number(r[r.length-1]),t]}return n&&t.parseNodeNameCache.set(e,i),i}function Dy(e,t,n){let r=J(`pad`,e,t,n);if(r===`explicit`){r=J(`explicitPaddings`,e,t,n);let i=[[0,0],[0,0],[0,0],[0,0]];for(let e=0;e<4;e++)i[e][0]=r[e*2],i[e][1]=r[e*2+1];return i}return r}function Oy(e){return e.kept?e:Po(e)}var ky=s({json:()=>Ay}),Ay=[{tfOpName:`Add`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`AddV2`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`AddN`,category:`arithmetic`,inputs:[{start:0,end:0,name:`tensors`,type:`tensors`}]},{tfOpName:`BiasAdd`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0},{tfName:`data_format`,name:`dataFormat`,type:`string`,notSupported:!0}]},{tfOpName:`Sub`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`RealDiv`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Div`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`DivNoNan`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`FloorDiv`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Mul`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Maximum`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Minimum`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Pow`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`SquaredDifference`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Mod`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`FloorMod`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]}],jy=s({json:()=>My}),My=[{tfOpName:`Abs`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Acos`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Asin`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Atan`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Atan2`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`y`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Ceil`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`ClipByValue`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`clipValueMin`,type:`number`},{start:2,name:`clipValueMax`,type:`number`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Complex`,category:`basic_math`,inputs:[{start:0,name:`real`,type:`tensor`},{start:1,name:`imag`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`ComplexAbs`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Cos`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Cosh`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Elu`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Exp`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Floor`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Log`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Imag`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0},{tfName:`Tout`,name:`outputType`,type:`dtype`,notSupported:!0}]},{tfOpName:`Neg`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Real`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0},{tfName:`Tout`,name:`outputType`,type:`dtype`,notSupported:!0}]},{tfOpName:`Prelu`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`alpha`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Relu`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Relu6`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Selu`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Sigmoid`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Sin`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Sinh`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Sqrt`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Rsqrt`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Square`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Tan`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Tanh`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Sign`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Round`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Expm1`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Log1p`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Reciprocal`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Softplus`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Asinh`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Acosh`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Atanh`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Erf`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`LeakyRelu`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`alpha`,name:`alpha`,type:`number`,defaultValue:.2},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`IsNan`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`IsFinite`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`IsInf`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]}],Ny=s({json:()=>Py}),Py=[{tfOpName:`EmptyTensorList`,category:`control`,inputs:[{start:0,name:`elementShape`,type:`shape`},{start:1,name:`maxNumElements`,type:`number`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`LoopCond`,category:`control`,inputs:[{start:0,name:`pred`,type:`tensor`}]},{tfOpName:`Switch`,category:`control`,inputs:[{start:0,name:`data`,type:`tensor`},{start:1,name:`pred`,type:`tensor`}]},{tfOpName:`Merge`,category:`control`,inputs:[{start:0,end:0,name:`tensors`,type:`tensors`}]},{tfOpName:`Enter`,category:`control`,inputs:[{start:0,name:`tensor`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0},{tfName:`frame_name`,name:`frameName`,type:`string`},{tfName:`is_constant`,name:`isConstant`,type:`bool`}]},{tfOpName:`Exit`,category:`control`,inputs:[{start:0,name:`tensor`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`NextIteration`,category:`control`,inputs:[{start:0,name:`tensor`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`TensorArrayV3`,category:`control`,inputs:[{start:0,name:`size`,type:`number`}],attrs:[{tfName:`dtype`,name:`dtype`,type:`dtype`},{tfName:`element_shape`,name:`elementShape`,type:`shape`},{tfName:`dynamic_size`,name:`dynamicSize`,type:`bool`},{tfName:`clear_after_read`,name:`clearAfterRead`,type:`bool`},{tfName:`identical_element_shapes`,name:`identicalElementShapes`,type:`bool`},{tfName:`tensor_array_name`,name:`name`,type:`string`}]},{tfOpName:`TensorArrayWriteV3`,category:`control`,inputs:[{start:0,name:`tensorArrayId`,type:`tensor`},{start:1,name:`index`,type:`number`},{start:2,name:`tensor`,type:`tensor`},{start:3,name:`flowIn`,type:`number`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`TensorArrayReadV3`,category:`control`,inputs:[{start:0,name:`tensorArrayId`,type:`tensor`},{start:1,name:`index`,type:`number`},{start:2,name:`flowIn`,type:`number`}],attrs:[{tfName:`dtype`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`TensorArrayGatherV3`,category:`control`,inputs:[{start:0,name:`tensorArrayId`,type:`tensor`},{start:1,name:`indices`,type:`number[]`},{start:2,name:`flowIn`,type:`number`}],attrs:[{tfName:`dtype`,name:`dtype`,type:`dtype`},{tfName:`element_shape`,name:`elementShape`,type:`shape`}]},{tfOpName:`TensorArrayScatterV3`,category:`control`,inputs:[{start:0,name:`tensorArrayId`,type:`tensor`},{start:1,name:`indices`,type:`number[]`},{start:2,name:`tensor`,type:`tensor`},{start:3,name:`flowIn`,type:`number`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`}]},{tfOpName:`TensorArrayConcatV3`,category:`control`,inputs:[{start:0,name:`tensorArrayId`,type:`tensor`},{start:1,name:`flowIn`,type:`number`}],attrs:[{tfName:`dtype`,name:`dtype`,type:`dtype`},{tfName:`element_shape_except0`,name:`elementShapeExcept0`,type:`shape`,notSupported:!0}]},{tfOpName:`TensorArraySplitV3`,category:`control`,inputs:[{start:0,name:`tensorArrayId`,type:`tensor`},{start:1,name:`tensor`,type:`tensor`},{start:2,name:`lengths`,type:`number[]`},{start:3,name:`flowIn`,type:`number`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`}]},{tfOpName:`TensorArraySizeV3`,category:`control`,inputs:[{start:0,name:`tensorArrayId`,type:`tensor`},{start:1,name:`flowIn`,type:`number`}]},{tfOpName:`TensorArrayCloseV3`,category:`control`,inputs:[{start:0,name:`tensorArrayId`,type:`tensor`}]},{tfOpName:`StatelessIf`,category:`control`,inputs:[{start:0,name:`cond`,type:`tensor`},{start:1,end:0,name:`args`,type:`tensors`}],attrs:[{tfName:`then_branch`,name:`thenBranch`,type:`func`},{tfName:`else_branch`,name:`elseBranch`,type:`func`}]},{tfOpName:`If`,category:`control`,inputs:[{start:0,name:`cond`,type:`tensor`},{start:1,end:0,name:`args`,type:`tensors`}],attrs:[{tfName:`then_branch`,name:`thenBranch`,type:`func`},{tfName:`else_branch`,name:`elseBranch`,type:`func`}]},{tfOpName:`StatelessWhile`,category:`control`,inputs:[{start:0,end:0,name:`args`,type:`tensors`}],attrs:[{tfName:`cond`,name:`cond`,type:`func`},{tfName:`body`,name:`body`,type:`func`}]},{tfOpName:`While`,category:`control`,inputs:[{start:0,end:0,name:`args`,type:`tensors`}],attrs:[{tfName:`cond`,name:`cond`,type:`func`},{tfName:`body`,name:`body`,type:`func`}]},{tfOpName:`TensorListScatter`,category:`control`,inputs:[{start:0,name:`tensor`,type:`tensor`},{start:1,name:`indices`,type:`number[]`},{start:2,name:`elementShape`,type:`shape`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListScatterV2`,category:`control`,inputs:[{start:0,name:`tensor`,type:`tensor`},{start:1,name:`indices`,type:`number[]`},{start:2,name:`elementShape`,type:`shape`},{start:3,name:`numElements`,type:`number`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListGather`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`},{start:1,name:`indices`,type:`number[]`},{start:2,name:`elementShape`,type:`shape`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListGetItem`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`},{start:1,name:`index`,type:`number`},{start:2,name:`elementShape`,type:`shape`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListSetItem`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`},{start:1,name:`index`,type:`number`},{start:2,name:`tensor`,type:`tensor`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListReserve`,category:`control`,inputs:[{start:0,name:`elementShape`,type:`shape`},{start:1,name:`numElements`,type:`number`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListFromTensor`,category:`control`,inputs:[{start:0,name:`tensor`,type:`tensor`},{start:1,name:`elementShape`,type:`shape`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListStack`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`},{start:1,name:`elementShape`,type:`shape`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`},{tfName:`num_elements`,name:`numElements`,type:`dtype`}]},{tfOpName:`TensorListSplit`,category:`control`,inputs:[{start:0,name:`tensor`,type:`tensor`},{start:1,name:`elementShape`,type:`shape`},{start:2,name:`lengths`,type:`number[]`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListConcat`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`}],attrs:[{tfName:`element_shape`,name:`elementShape`,type:`shape`},{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListConcatV2`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`}],attrs:[{tfName:`element_shape`,name:`elementShape`,type:`shape`},{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListPopBack`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`},{start:1,name:`elementShape`,type:`shape`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListPushBack`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`},{start:1,name:`tensor`,type:`tensor`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListLength`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`}]},{tfOpName:`TensorListResize`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`},{start:1,name:`size`,type:`number`}]}],Fy=s({json:()=>Iy}),Iy=[{tfOpName:`AvgPool`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,notSupported:!0},{tfName:`ksize`,name:`kernelSize`,type:`number[]`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`MaxPool`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,notSupported:!0},{tfName:`ksize`,name:`kernelSize`,type:`number[]`},{tfName:`explicit_paddings`,name:`explicitPaddings`,type:`number[]`,defaultValue:[],notSupported:!0},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`MaxPoolWithArgmax`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`ksize`,name:`kernelSize`,type:`number[]`},{tfName:`include_batch_in_index`,name:`includeBatchInIndex`,type:`bool`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`AvgPool3D`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,notSupported:!0},{tfName:`ksize`,name:`kernelSize`,type:`number[]`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`MaxPool3D`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,notSupported:!0},{tfName:`ksize`,name:`kernelSize`,type:`number[]`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Conv1D`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`filter`,type:`tensor`}],attrs:[{tfName:`stride`,name:`stride`,type:`number`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,defaultValue:`NWC`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0},{tfName:`dilation`,name:`dilation`,type:`number`,defaultValue:1}]},{tfOpName:`Conv2D`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`filter`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0},{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`useCudnnOnGpu`,name:`useCudnnOnGpu`,type:`bool`},{tfName:`data_format`,name:`dataFormat`,type:`string`,defaultValue:`NHWC`},{tfName:`explicit_paddings`,name:`explicitPaddings`,type:`number[]`,defaultValue:[]},{tfName:`dilations`,name:`dilations`,type:`number[]`}]},{tfOpName:`_FusedConv2D`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`filter`,type:`tensor`},{start:2,end:0,name:`args`,type:`tensors`}],attrs:[{tfName:`num_args`,name:`numArgs`,type:`number`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0},{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`explicit_paddings`,name:`explicitPaddings`,type:`number[]`,defaultValue:[]},{tfName:`use_cudnn_on_gpu`,name:`useCudnnOnGpu`,type:`bool`,defaultValue:!0},{tfName:`data_format`,name:`dataFormat`,type:`string`,defaultValue:`NHWC`},{tfName:`dilations`,name:`dilations`,type:`number[]`,defaultValue:[1,1,1,1]},{tfName:`fused_ops`,name:`fusedOps`,type:`string[]`,defaultValue:[]},{tfName:`epsilon`,name:`epsilon`,type:`number`,defaultValue:1e-4},{tfName:`leakyrelu_alpha`,name:`leakyreluAlpha`,type:`number`,defaultValue:.2}]},{tfOpName:`Conv2DBackpropInput`,category:`convolution`,inputs:[{start:2,name:`x`,type:`tensor`},{start:1,name:`filter`,type:`tensor`},{start:0,name:`outputShape`,type:`number[]`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,notSupported:!0},{tfName:`explicit_paddings`,name:`explicitPaddings`,type:`number[]`,defaultValue:[]},{tfName:`dilations`,name:`dilations`,type:`number[]`,notSupported:!0}]},{tfOpName:`DepthwiseConv2d`,category:`convolution`,inputs:[{start:0,name:`input`,type:`tensor`},{start:1,name:`filter`,type:`tensor`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,defaultValue:`NHWC`},{tfName:`explicit_paddings`,name:`explicitPaddings`,type:`number[]`,defaultValue:[]},{tfName:`dilations`,name:`dilations`,type:`number[]`}]},{tfOpName:`DepthwiseConv2dNative`,category:`convolution`,inputs:[{start:0,name:`input`,type:`tensor`},{start:1,name:`filter`,type:`tensor`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,defaultValue:`NHWC`},{tfName:`explicit_paddings`,name:`explicitPaddings`,type:`number[]`,defaultValue:[]},{tfName:`dilations`,name:`dilations`,type:`number[]`}]},{tfOpName:`FusedDepthwiseConv2dNative`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`filter`,type:`tensor`},{start:2,end:0,name:`args`,type:`tensors`}],attrs:[{tfName:`num_args`,name:`numArgs`,type:`number`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0},{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,defaultValue:`NHWC`},{tfName:`dilations`,name:`dilations`,type:`number[]`,defaultValue:[1,1,1,1]},{tfName:`fused_ops`,name:`fusedOps`,type:`string[]`,defaultValue:[]},{tfName:`explicit_paddings`,name:`explicitPaddings`,type:`number[]`,defaultValue:[]}]},{tfOpName:`Conv3D`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`filter`,type:`tensor`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,defaultValue:`NHWC`},{tfName:`dilations`,name:`dilations`,type:`number[]`}]},{tfOpName:`Dilation2D`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`filter`,type:`tensor`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`rates`,name:`dilations`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`}]}],Ly=s({json:()=>Ry}),Ry=[{tfOpName:`Fill`,category:`creation`,inputs:[{start:0,name:`shape`,type:`number[]`},{start:1,name:`value`,type:`number`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`}]},{tfOpName:`LinSpace`,category:`creation`,inputs:[{start:0,name:`start`,type:`number`},{start:1,name:`stop`,type:`number`},{start:2,name:`num`,type:`number`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`OneHot`,category:`creation`,inputs:[{start:0,name:`indices`,type:`tensor`},{start:1,name:`depth`,type:`number`},{start:2,name:`onValue`,type:`number`,defaultValue:1},{start:3,name:`offValue`,type:`number`,defaultValue:0}],attrs:[{tfName:`axis`,name:`axis`,type:`number`,notSupported:!0},{tfName:`T`,name:`dtype`,type:`dtype`}]},{tfOpName:`Ones`,category:`creation`,inputs:[{start:0,name:`shape`,type:`number[]`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`}]},{tfOpName:`OnesLike`,category:`creation`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`dtype`,name:`dtype`,type:`dtype`}]},{tfOpName:`RandomStandardNormal`,category:`creation`,inputs:[{start:0,name:`shape`,type:`number[]`}],attrs:[{tfName:`seed`,name:`seed`,type:`number`,defaultValue:0},{tfName:`seed2`,name:`seed2`,type:`number`,defaultValue:0,notSupported:!0},{tfName:`dtype`,name:`dtype`,type:`dtype`},{tfName:`T`,name:`T`,type:`number`,notSupported:!0}]},{tfOpName:`RandomUniform`,category:`creation`,inputs:[{start:0,name:`shape`,type:`number[]`}],attrs:[{tfName:`minval`,name:`minval`,type:`number`,defaultValue:0},{tfName:`maxval`,name:`maxval`,type:`number`,defaultValue:1},{tfName:`dtype`,name:`dtype`,type:`dtype`},{tfName:`seed`,name:`seed`,type:`number`,defaultValue:0},{tfName:`seed2`,name:`seed2`,type:`number`,defaultValue:0,notSupported:!0},{tfName:`T`,name:`T`,type:`number`,notSupported:!0}]},{tfOpName:`RandomUniformInt`,category:`creation`,inputs:[{start:0,name:`shape`,type:`number[]`}],attrs:[{tfName:`minval`,name:`minval`,type:`number`},{tfName:`maxval`,name:`maxval`,type:`number`},{tfName:`seed`,name:`seed`,type:`number`,defaultValue:0},{tfName:`seed2`,name:`seed2`,type:`number`,defaultValue:0,notSupported:!0}]},{tfOpName:`Range`,category:`creation`,inputs:[{start:0,name:`start`,type:`number`},{start:1,name:`stop`,type:`number`},{start:2,name:`step`,type:`number`,defaultValue:0}],attrs:[{tfName:`Tidx`,name:`dtype`,type:`dtype`}]},{tfOpName:`TruncatedNormal`,category:`creation`,inputs:[{start:0,name:`shape`,type:`number[]`}],attrs:[{tfName:`means`,name:`mean`,type:`number`,defaultValue:0},{tfName:`stddev`,name:`stdDev`,type:`number`,defaultValue:1},{tfName:`seed`,name:`seed`,type:`number`},{tfName:`seed2`,name:`seed2`,type:`number`,defaultValue:0,notSupported:!0},{tfName:`dtype`,name:`dtype`,type:`dtype`},{tfName:`T`,name:`T`,type:`number`,notSupported:!0}]},{tfOpName:`Zeros`,category:`creation`,inputs:[{start:0,name:`shape`,type:`number[]`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`}]},{tfOpName:`ZerosLike`,category:`creation`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`}]},{tfOpName:`Multinomial`,category:`creation`,inputs:[{start:0,name:`logits`,type:`tensor`},{start:1,name:`numSamples`,type:`number`}],attrs:[{tfName:`seed`,name:`seed`,type:`number`},{tfName:`seed2`,name:`seed2`,type:`number`},{tfName:`T`,name:`dtype`,type:`dtype`},{tfName:`output_dtype`,name:`output_dtype`,type:`dtype`}]}],zy=s({json:()=>By}),By=[{tfOpName:`NonMaxSuppressionV2`,category:`dynamic`,inputs:[{start:0,name:`boxes`,type:`tensor`},{start:1,name:`scores`,type:`tensor`},{start:2,name:`maxOutputSize`,type:`number`},{start:3,name:`iouThreshold`,type:`number`}]},{tfOpName:`NonMaxSuppressionV3`,category:`dynamic`,inputs:[{start:0,name:`boxes`,type:`tensor`},{start:1,name:`scores`,type:`tensor`},{start:2,name:`maxOutputSize`,type:`number`},{start:3,name:`iouThreshold`,type:`number`},{start:4,name:`scoreThreshold`,type:`number`}]},{tfOpName:`NonMaxSuppressionV4`,category:`dynamic`,inputs:[{start:0,name:`boxes`,type:`tensor`},{start:1,name:`scores`,type:`tensor`},{start:2,name:`maxOutputSize`,type:`number`},{start:3,name:`iouThreshold`,type:`number`},{start:4,name:`scoreThreshold`,type:`number`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0},{tfName:`T_threshold`,name:`threshold`,type:`dtype`,notSupported:!0},{tfName:`pad_to_max_output_size`,name:`padToMaxOutputSize`,type:`bool`}]},{tfOpName:`NonMaxSuppressionV5`,category:`dynamic`,inputs:[{start:0,name:`boxes`,type:`tensor`},{start:1,name:`scores`,type:`tensor`},{start:2,name:`maxOutputSize`,type:`number`},{start:3,name:`iouThreshold`,type:`number`},{start:4,name:`scoreThreshold`,type:`number`},{start:5,name:`softNmsSigma`,type:`number`}]},{tfOpName:`Where`,category:`dynamic`,inputs:[{start:0,name:`condition`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`ListDiff`,category:`dynamic`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`y`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]}],Vy=s({json:()=>Hy}),Hy=[{tfOpName:`LowerBound`,category:`evaluation`,inputs:[{start:0,name:`sortedSequence`,type:`tensor`},{start:1,name:`values`,type:`tensor`}]},{tfOpName:`TopKV2`,category:`evaluation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`k`,type:`number`}],attrs:[{tfName:`sorted`,name:`sorted`,type:`bool`}]},{tfOpName:`UpperBound`,category:`evaluation`,inputs:[{start:0,name:`sortedSequence`,type:`tensor`},{start:1,name:`values`,type:`tensor`}]},{tfOpName:`Unique`,category:`evaluation`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`UniqueV2`,category:`evaluation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number`}]}],Uy=s({json:()=>Wy}),Wy=[{tfOpName:`PlaceholderWithDefault`,category:`graph`,inputs:[{start:0,name:`default`,type:`tensor`}],attrs:[{tfName:`shape`,name:`shape`,type:`shape`},{tfName:`dtype`,name:`dtype`,type:`dtype`}]},{tfOpName:`Placeholder`,category:`graph`,attrs:[{tfName:`shape`,name:`shape`,type:`shape`},{tfName:`dtype`,name:`dtype`,type:`dtype`}]},{tfOpName:`Const`,category:`graph`},{tfOpName:`Identity`,category:`graph`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`IdentityN`,category:`graph`,inputs:[{start:0,end:0,name:`x`,type:`tensors`}]},{tfOpName:`Snapshot`,category:`graph`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`Rank`,category:`graph`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`Size`,category:`graph`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`Shape`,category:`graph`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`ShapeN`,category:`graph`,inputs:[{start:0,end:0,name:`x`,type:`tensors`}]},{tfOpName:`Print`,category:`graph`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`data`,type:`tensors`}],attrs:[{tfName:`message`,name:`message`,type:`string`},{tfName:`first_n`,name:`firstN`,type:`number`,notSupported:!0},{tfName:`summarize`,name:`summarize`,type:`number`,defaultValue:3}]},{tfOpName:`NoOp`,category:`graph`,inputs:[]},{tfOpName:`StopGradient`,category:`graph`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`FakeQuantWithMinMaxVars`,category:`graph`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`min`,name:`min`,type:`number`},{tfName:`max`,name:`max`,type:`number`}]}],Gy=s({json:()=>Ky}),Ky=[{tfOpName:`HashTable`,category:`hash_table`,inputs:[],attrs:[{tfName:`shared_name`,name:`sharedName`,type:`string`},{tfName:`use_node_name_sharing`,name:`useNodeNameSharing`,type:`bool`},{tfName:`key_dtype`,name:`keyDType`,type:`dtype`},{tfName:`value_dtype`,name:`valueDType`,type:`dtype`}]},{tfOpName:`HashTableV2`,category:`hash_table`,inputs:[],attrs:[{tfName:`shared_name`,name:`sharedName`,type:`string`},{tfName:`use_node_name_sharing`,name:`useNodeNameSharing`,type:`bool`},{tfName:`key_dtype`,name:`keyDType`,type:`dtype`},{tfName:`value_dtype`,name:`valueDType`,type:`dtype`}]},{tfOpName:`LookupTableImport`,category:`hash_table`,inputs:[{start:0,name:`tableHandle`,type:`tensor`},{start:1,name:`keys`,type:`tensor`},{start:2,name:`values`,type:`tensor`}],attrs:[{tfName:`Tin`,name:`tIn`,type:`dtype`,notSupported:!0},{tfName:`Tout`,name:`tOut`,type:`dtype`,notSupported:!0}]},{tfOpName:`LookupTableImportV2`,category:`hash_table`,inputs:[{start:0,name:`tableHandle`,type:`tensor`},{start:1,name:`keys`,type:`tensor`},{start:2,name:`values`,type:`tensor`}],attrs:[{tfName:`Tin`,name:`tIn`,type:`dtype`,notSupported:!0},{tfName:`Tout`,name:`tOut`,type:`dtype`,notSupported:!0}]},{tfOpName:`LookupTableFind`,category:`hash_table`,inputs:[{start:0,name:`tableHandle`,type:`tensor`},{start:1,name:`keys`,type:`tensor`},{start:2,name:`defaultValue`,type:`tensor`}],attrs:[{tfName:`Tin`,name:`tIn`,type:`dtype`,notSupported:!0},{tfName:`Tout`,name:`tOut`,type:`dtype`,notSupported:!0}]},{tfOpName:`LookupTableFindV2`,category:`hash_table`,inputs:[{start:0,name:`tableHandle`,type:`tensor`},{start:1,name:`keys`,type:`tensor`},{start:2,name:`defaultValue`,type:`tensor`}],attrs:[{tfName:`Tin`,name:`tIn`,type:`dtype`,notSupported:!0},{tfName:`Tout`,name:`tOut`,type:`dtype`,notSupported:!0}]},{tfOpName:`LookupTableSize`,category:`hash_table`,inputs:[{start:0,name:`tableHandle`,type:`tensor`}]},{tfOpName:`LookupTableSizeV2`,category:`hash_table`,inputs:[{start:0,name:`tableHandle`,type:`tensor`}]},{tfOpName:`InitializeTable`,category:`hash_table`,inputs:[{start:0,name:`tableHandle`,type:`tensor`},{start:1,name:`keys`,type:`tensor`},{start:2,name:`values`,type:`tensor`}]},{tfOpName:`InitializeTableV2`,category:`hash_table`,inputs:[{start:0,name:`tableHandle`,type:`tensor`},{start:1,name:`keys`,type:`tensor`},{start:2,name:`values`,type:`tensor`}]}],qy=s({json:()=>Jy}),Jy=[{tfOpName:`ResizeBilinear`,category:`image`,inputs:[{start:0,name:`images`,type:`tensor`},{start:1,name:`size`,type:`number[]`}],attrs:[{tfName:`align_corners`,name:`alignCorners`,type:`bool`},{tfName:`half_pixel_centers`,name:`halfPixelCenters`,type:`bool`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`ResizeNearestNeighbor`,category:`image`,inputs:[{start:0,name:`images`,type:`tensor`},{start:1,name:`size`,type:`number[]`}],attrs:[{tfName:`align_corners`,name:`alignCorners`,type:`bool`},{tfName:`half_pixel_centers`,name:`halfPixelCenters`,type:`bool`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`CropAndResize`,category:`image`,inputs:[{start:0,name:`image`,type:`tensor`},{start:1,name:`boxes`,type:`tensor`},{start:2,name:`boxInd`,type:`tensor`},{start:3,name:`cropSize`,type:`number[]`}],attrs:[{tfName:`method`,name:`method`,type:`string`},{tfName:`extrapolation_value`,name:`extrapolationValue`,type:`number`}]},{tfOpName:`ImageProjectiveTransformV3`,category:`image`,inputs:[{start:0,name:`images`,type:`tensor`},{start:1,name:`transforms`,type:`tensor`},{start:2,name:`outputShape`,type:`number[]`},{start:3,name:`fillValue`,type:`number`}],attrs:[{tfName:`interpolation`,name:`interpolation`,type:`string`},{tfName:`fill_mode`,name:`fillMode`,type:`string`}]}],Yy=s({json:()=>Xy}),Xy=[{tfOpName:`Equal`,category:`logical`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`NotEqual`,category:`logical`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Greater`,category:`logical`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`GreaterEqual`,category:`logical`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Less`,category:`logical`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`LessEqual`,category:`logical`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`LogicalAnd`,category:`logical`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`LogicalNot`,category:`logical`,inputs:[{start:0,name:`a`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`LogicalOr`,category:`logical`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Select`,category:`logical`,inputs:[{start:0,name:`condition`,type:`tensor`},{start:1,name:`a`,type:`tensor`},{start:2,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`SelectV2`,category:`logical`,inputs:[{start:0,name:`condition`,type:`tensor`},{start:1,name:`a`,type:`tensor`},{start:2,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`BitwiseAnd`,category:`logical`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`y`,type:`tensor`}]}],Zy=s({json:()=>Qy}),Qy=[{tfOpName:`_FusedMatMul`,category:`matrices`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`},{start:2,end:0,name:`args`,type:`tensors`}],attrs:[{tfName:`num_args`,name:`numArgs`,type:`number`},{tfName:`fused_ops`,name:`fusedOps`,type:`string[]`,defaultValue:[]},{tfName:`epsilon`,name:`epsilon`,type:`number`,defaultValue:1e-4},{tfName:`transpose_a`,name:`transposeA`,type:`bool`,defaultValue:!1},{tfName:`transpose_b`,name:`transposeB`,type:`bool`,defaultValue:!1},{tfName:`leakyrelu_alpha`,name:`leakyreluAlpha`,type:`number`,defaultValue:.2},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`MatMul`,category:`matrices`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`transpose_a`,name:`transposeA`,type:`bool`,defaultValue:!1},{tfName:`transpose_b`,name:`transposeB`,type:`bool`,defaultValue:!1},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`BatchMatMul`,category:`matrices`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`adj_x`,name:`transposeA`,type:`bool`,defaultValue:!1},{tfName:`adj_y`,name:`transposeB`,type:`bool`,defaultValue:!1},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`BatchMatMulV2`,category:`matrices`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`adj_x`,name:`transposeA`,type:`bool`,defaultValue:!1},{tfName:`adj_y`,name:`transposeB`,type:`bool`,defaultValue:!1},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Transpose`,category:`matrices`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`perm`,type:`number[]`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Einsum`,category:`matrices`,inputs:[{start:0,end:0,name:`tensors`,type:`tensors`}],attrs:[{tfName:`equation`,name:`equation`,type:`string`},{tfName:`N`,name:`n`,type:`number`,defaultValue:2},{tfName:`T`,name:`dtype`,type:`dtype`}]},{tfOpName:`MatrixBandPart`,category:`matrices`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`numLower`,type:`tensor`},{start:1,name:`numUpper`,type:`tensor`}]}],$y=s({json:()=>eb}),eb=[{tfOpName:`EuclideanNorm`,category:`normalization`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number[]`}],attrs:[{tfName:`keep_dims`,name:`keepDims`,type:`bool`,defaultValue:!1}]},{tfOpName:`FusedBatchNorm`,category:`normalization`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`scale`,type:`tensor`},{start:2,name:`offset`,type:`tensor`},{start:3,name:`mean`,type:`tensor`},{start:4,name:`variance`,type:`tensor`}],attrs:[{tfName:`epsilon`,name:`epsilon`,type:`number`,defaultValue:.001},{tfName:`data_format`,name:`dataFormat`,type:`string`,notSupported:!0}]},{tfOpName:`FusedBatchNormV2`,category:`normalization`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`scale`,type:`tensor`},{start:2,name:`offset`,type:`tensor`},{start:3,name:`mean`,type:`tensor`},{start:4,name:`variance`,type:`tensor`}],attrs:[{tfName:`epsilon`,name:`epsilon`,type:`number`,defaultValue:.001},{tfName:`data_format`,name:`dataFormat`,type:`string`,notSupported:!0}]},{tfOpName:`FusedBatchNormV3`,category:`normalization`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`scale`,type:`tensor`},{start:2,name:`offset`,type:`tensor`},{start:3,name:`mean`,type:`tensor`},{start:4,name:`variance`,type:`tensor`}],attrs:[{tfName:`epsilon`,name:`epsilon`,type:`number`,defaultValue:.001},{tfName:`data_format`,name:`dataFormat`,type:`string`,notSupported:!0}]},{tfOpName:`LRN`,category:`normalization`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`depth_radius`,name:`radius`,type:`number`,defaultValue:5},{tfName:`bias`,name:`bias`,type:`number`,defaultValue:1},{tfName:`alpha`,name:`alpha`,type:`number`,defaultValue:1},{tfName:`beta`,name:`beta`,type:`number`,defaultValue:.5}]},{tfOpName:`Softmax`,category:`normalization`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`LogSoftmax`,category:`normalization`,inputs:[{start:0,name:`x`,type:`tensor`}]}],tb=s({json:()=>nb}),nb=[{tfOpName:`Bincount`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`size`,type:`number`},{start:2,name:`weights`,type:`tensor`}]},{tfOpName:`DenseBincount`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`size`,type:`number`},{start:2,name:`weights`,type:`tensor`}],attrs:[{tfName:`binary_output`,name:`binaryOutput`,type:`bool`}]},{tfOpName:`Max`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number[]`}],attrs:[{tfName:`keep_dims`,name:`keepDims`,type:`bool`}]},{tfOpName:`Mean`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number[]`}],attrs:[{tfName:`keep_dims`,name:`keepDims`,type:`bool`}]},{tfOpName:`Min`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number[]`}],attrs:[{tfName:`keep_dims`,name:`keepDims`,type:`bool`}]},{tfOpName:`Sum`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number[]`}],attrs:[{tfName:`keep_dims`,name:`keepDims`,type:`bool`}]},{tfOpName:`All`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number[]`}],attrs:[{tfName:`keep_dims`,name:`keepDims`,type:`bool`}]},{tfOpName:`Any`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number[]`}],attrs:[{tfName:`keep_dims`,name:`keepDims`,type:`bool`}]},{tfOpName:`ArgMax`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number`}]},{tfOpName:`ArgMin`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number`}]},{tfOpName:`Prod`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number[]`}],attrs:[{tfName:`keep_dims`,name:`keepDims`,type:`bool`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Cumprod`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number`}],attrs:[{tfName:`exclusive`,name:`exclusive`,type:`bool`},{tfName:`reverse`,name:`reverse`,type:`bool`}]},{tfOpName:`Cumsum`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number`}],attrs:[{tfName:`exclusive`,name:`exclusive`,type:`bool`},{tfName:`reverse`,name:`reverse`,type:`bool`}]}],rb=s({json:()=>ib}),ib=[{tfOpName:`ConcatV2`,category:`slice_join`,inputs:[{start:0,end:-1,name:`tensors`,type:`tensors`},{start:-1,name:`axis`,type:`number`}],attrs:[{tfName:`N`,name:`n`,type:`number`,defaultValue:2}]},{tfOpName:`Concat`,category:`slice_join`,inputs:[{start:1,end:0,name:`tensors`,type:`tensors`},{start:0,name:`axis`,type:`number`}],attrs:[{tfName:`N`,name:`n`,type:`number`,defaultValue:2}]},{tfOpName:`GatherV2`,category:`slice_join`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`indices`,type:`tensor`},{start:2,name:`axis`,type:`number`,defaultValue:0}],attrs:[{tfName:`batch_dims`,name:`batchDims`,type:`number`,defaultValue:0}]},{tfOpName:`Gather`,category:`slice_join`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`indices`,type:`tensor`}],attrs:[{tfName:`validate_indices`,name:`validateIndices`,type:`bool`,notSupported:!0}]},{tfOpName:`Reverse`,category:`slice_join`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`dims`,type:`bool[]`}]},{tfOpName:`ReverseV2`,category:`slice_join`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number[]`}]},{tfOpName:`Slice`,category:`slice_join`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`begin`,type:`number[]`},{start:2,name:`size`,type:`number[]`}]},{tfOpName:`StridedSlice`,category:`slice_join`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`begin`,type:`number[]`},{start:2,name:`end`,type:`number[]`},{start:3,name:`strides`,type:`number[]`}],attrs:[{tfName:`begin_mask`,name:`beginMask`,type:`number`,defaultValue:0},{tfName:`end_mask`,name:`endMask`,type:`number`,defaultValue:0},{tfName:`new_axis_mask`,name:`newAxisMask`,type:`number`,defaultValue:0},{tfName:`ellipsis_mask`,name:`ellipsisMask`,type:`number`,defaultValue:0},{tfName:`shrink_axis_mask`,name:`shrinkAxisMask`,type:`number`,defaultValue:0}]},{tfOpName:`Pack`,category:`slice_join`,inputs:[{start:0,end:0,name:`tensors`,type:`tensors`}],attrs:[{tfName:`axis`,name:`axis`,type:`number`,defaultValue:0}]},{tfOpName:`Unpack`,category:`slice_join`,inputs:[{start:0,name:`tensor`,type:`tensor`}],attrs:[{tfName:`axis`,name:`axis`,type:`number`,defaultValue:0},{tfName:`num`,name:`num`,type:`number`,defaultValue:0,notSupported:!0}]},{tfOpName:`Tile`,category:`slice_join`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`reps`,type:`number[]`}]},{tfOpName:`Split`,category:`slice_join`,inputs:[{start:0,name:`axis`,type:`number`,defaultValue:0},{start:1,name:`x`,type:`tensor`}],attrs:[{tfName:`num_split`,name:`numOrSizeSplits`,type:`number`,defaultValue:1}]},{tfOpName:`SplitV`,category:`slice_join`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`numOrSizeSplits`,type:`number[]`},{start:2,name:`axis`,type:`number`,defaultValue:0}]},{tfOpName:`ScatterNd`,category:`slice_join`,inputs:[{start:0,name:`indices`,type:`tensor`},{start:1,name:`values`,type:`tensor`},{start:2,name:`shape`,type:`number[]`}]},{tfOpName:`GatherNd`,category:`slice_join`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`indices`,type:`tensor`}]},{tfOpName:`SparseToDense`,category:`slice_join`,inputs:[{start:0,name:`sparseIndices`,type:`tensor`},{start:1,name:`outputShape`,type:`number[]`},{start:2,name:`sparseValues`,type:`tensor`},{start:3,name:`defaultValue`,type:`tensor`}],attrs:[{tfName:`validate_indices`,name:`validateIndices`,type:`bool`,defaultValue:!1,notSupported:!0}]},{tfOpName:`TensorScatterUpdate`,category:`slice_join`,inputs:[{start:0,name:`tensor`,type:`tensor`},{start:1,name:`indices`,type:`tensor`},{start:2,name:`values`,type:`tensor`}]}],ab=s({json:()=>ob}),ob=[{tfOpName:`SparseFillEmptyRows`,category:`sparse`,inputs:[{start:0,name:`indices`,type:`tensor`},{start:1,name:`values`,type:`tensor`},{start:2,name:`denseShape`,type:`tensor`},{start:3,name:`defaultValue`,type:`tensor`}]},{tfOpName:`SparseReshape`,category:`sparse`,inputs:[{start:0,name:`inputIndices`,type:`tensor`},{start:1,name:`inputShape`,type:`tensor`},{start:2,name:`newShape`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`SparseSegmentMean`,category:`sparse`,inputs:[{start:0,name:`data`,type:`tensor`},{start:1,name:`indices`,type:`tensor`},{start:2,name:`segmentIds`,type:`tensor`}]},{tfOpName:`SparseSegmentSum`,category:`sparse`,inputs:[{start:0,name:`data`,type:`tensor`},{start:1,name:`indices`,type:`tensor`},{start:2,name:`segmentIds`,type:`tensor`}]}],sb=s({json:()=>cb}),cb=[{tfOpName:`FFT`,category:`spectral`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`IFFT`,category:`spectral`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`RFFT`,category:`spectral`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`fft_length`,type:`number`,notSupported:!0}]},{tfOpName:`IRFFT`,category:`spectral`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`fft_length`,type:`number`,notSupported:!0}]}],lb=s({json:()=>ub}),ub=[{tfOpName:`StaticRegexReplace`,category:`string`,inputs:[{start:0,name:`input`,type:`tensor`}],attrs:[{tfName:`pattern`,name:`pattern`,type:`string`},{tfName:`rewrite`,name:`rewrite`,type:`string`},{tfName:`replace_global`,name:`replaceGlobal`,type:`bool`}]},{tfOpName:`StringNGrams`,category:`string`,inputs:[{start:0,name:`data`,type:`tensor`},{start:1,name:`dataSplits`,type:`tensor`}],attrs:[{tfName:`separator`,name:`separator`,type:`string`},{tfName:`ngram_widths`,name:`nGramWidths`,type:`number[]`},{tfName:`left_pad`,name:`leftPad`,type:`string`},{tfName:`right_pad`,name:`rightPad`,type:`string`},{tfName:`pad_width`,name:`padWidth`,type:`number`},{tfName:`preserve_short_sequences`,name:`preserveShortSequences`,type:`bool`}],outputs:[`ngrams`,`ngrams_splits`]},{tfOpName:`StringSplit`,category:`string`,inputs:[{start:0,name:`input`,type:`tensor`},{start:1,name:`delimiter`,type:`tensor`}],attrs:[{tfName:`skip_empty`,name:`skipEmpty`,type:`bool`}],outputs:[`indices`,`values`,`shape`]},{tfOpName:`StringToHashBucketFast`,category:`string`,inputs:[{start:0,name:`input`,type:`tensor`}],attrs:[{tfName:`num_buckets`,name:`numBuckets`,type:`number`}]}],db=s({json:()=>fb}),fb=[{tfOpName:`Cast`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`SrcT`,name:`sdtype`,type:`dtype`,notSupported:!0},{tfName:`DstT`,name:`dtype`,type:`dtype`}]},{tfOpName:`ExpandDims`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number`}]},{tfOpName:`MirrorPad`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`padding`,type:`number[]`}],attrs:[{tfName:`mode`,name:`mode`,type:`string`}]},{tfOpName:`Pad`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`padding`,type:`number[]`}],attrs:[{tfName:`constant_value`,name:`constantValue`,type:`number`,defaultValue:0}]},{tfOpName:`PadV2`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`padding`,type:`number[]`},{start:2,name:`constantValue`,type:`number`,defaultValue:0}]},{tfOpName:`Reshape`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`shape`,type:`number[]`}]},{tfOpName:`EnsureShape`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`shape`,type:`number[]`}]},{tfOpName:`Squeeze`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`axis`,tfDeprecatedName:`squeeze_dims`,name:`axis`,type:`number[]`}]},{tfOpName:`SpaceToBatchND`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`blockShape`,type:`number[]`},{start:2,name:`paddings`,type:`number[]`}]},{tfOpName:`BatchToSpaceND`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`blockShape`,type:`number[]`},{start:2,name:`crops`,type:`number[]`}]},{tfOpName:`DepthToSpace`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`block_size`,name:`blockSize`,type:`number`},{tfName:`data_format`,name:`dataFormat`,type:`string`}]},{tfOpName:`BroadcastTo`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`shape`,type:`number[]`}],attrs:[]},{tfOpName:`BroadcastArgs`,category:`transformation`,inputs:[{start:0,name:`s0`,type:`tensor`},{start:1,name:`s1`,type:`tensor`}],attrs:[]}],pb=class{static get Instance(){return this._instance||=new this}constructor(){let e=[ky,jy,Ny,Fy,Ly,zy,Vy,Uy,Gy,qy,Yy,Zy,$y,tb,rb,ab,sb,lb,db],t=[].concat(...e.map(e=>e.json));this.opMappers=t.reduce((e,t)=>(e[t.tfOpName]=t,e),{})}transformGraph(e,t={}){let n=e.node,r=[],i=[],a=[],o=n.reduce((e,t)=>(e[t.name]=this.mapNode(t),t.op.startsWith(`Placeholder`)?r.push(e[t.name]):t.op===`Const`?i.push(e[t.name]):(t.input==null||t.input.length===0)&&a.push(e[t.name]),e),{}),s=[],c=[],l={},u={};t!=null&&(l=this.mapSignatureEntries(t.inputs),u=this.mapSignatureEntries(t.outputs));let d=Object.keys(o);d.forEach(e=>{let t=o[e];t.inputNames.forEach((e,n)=>{let[r,,i]=wy(e),a=o[r];if(a.outputs!=null){let e=a.outputs.indexOf(i);if(e!==-1){let i=`${r}:${e}`;t.inputNames[n]=i}}t.inputs.push(a),a.children.push(t)})}),Object.keys(u).length===0?d.forEach(e=>{let t=o[e];t.children.length===0&&c.push(t)}):Object.keys(u).forEach(e=>{let[t]=wy(e),n=o[t];n!=null&&(n.signatureKey=u[e],c.push(n))}),Object.keys(l).length>0?Object.keys(l).forEach(e=>{let[t]=wy(e),n=o[t];n&&(n.signatureKey=l[e],s.push(n))}):s=r;let f={};e.library!=null&&e.library.function!=null&&(f=e.library.function.reduce((e,t)=>(e[t.signature.name]=this.mapFunction(t),e),{}));let p={nodes:o,inputs:s,outputs:c,weights:i,placeholders:r,signature:t,functions:f};return a.length>0&&(p.initNodes=a),p}mapSignatureEntries(e){return Object.keys(e||{}).reduce((t,n)=>(t[e[n].name]=n,t),{})}mapNode(e){let t=xy(e.op)||this.opMappers[e.op]||{};e.attr??={};let n={name:e.name,op:e.op,category:t.category,inputNames:(e.input||[]).map(e=>e.startsWith(`^`)?e.slice(1):e),inputs:[],children:[],inputParams:{},attrParams:{},rawAttrs:e.attr,outputs:t.outputs};return t.inputs!=null&&(n.inputParams=t.inputs.reduce((e,t)=>(e[t.name]={type:t.type,inputIndexStart:t.start,inputIndexEnd:t.end},e),{})),t.attrs!=null&&(n.attrParams=t.attrs.reduce((t,n)=>{let r=n.type,i;switch(n.type){case`string`:i=gb(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=gb(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`string[]`:i=Eb(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=Eb(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`number`:i=vb(e.attr,n.tfName,n.defaultValue||0),i===void 0&&n.tfDeprecatedName&&(i=vb(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`number[]`:i=Tb(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=Tb(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`bool`:i=_b(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=_b(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`bool[]`:i=Ob(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=Ob(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`shape`:i=wb(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=wb(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`shape[]`:i=Db(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=Db(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`dtype`:i=xb(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=xb(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`dtype[]`:i=Sb(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=Sb(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`func`:i=bb(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=bb(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`tensor`:case`tensors`:break;default:throw Error(`Unsupported param type: ${n.type} for op: ${e.op}`)}return t[n.name]={value:i,type:r},t},{})),n}mapFunction(e){let t=e.nodeDef,n=[],r=[],i={};t!=null&&(i=t.reduce((e,t)=>(e[t.name]=this.mapNode(t),t.op===`Const`&&r.push(e[t.name]),e),{}));let a=[],o=[];e.signature.inputArg.forEach(e=>{let[t]=wy(e.name),n={name:t,op:`Placeholder`,inputs:[],inputNames:[],category:`graph`,inputParams:{},attrParams:{dtype:{value:yb(e.type),type:`dtype`}},children:[]};n.signatureKey=e.name,a.push(n),i[t]=n}),Object.keys(i).forEach(e=>{let t=i[e];t.inputNames.forEach((e,n)=>{let[r,,a]=wy(e),o=i[r];if(o.outputs!=null){let e=o.outputs.indexOf(a);if(e!==-1){let i=`${r}:${e}`;t.inputNames[n]=i}}t.inputs.push(o),o.children.push(t)})});let s=e.ret;e.signature.outputArg.forEach(e=>{let[t,n]=wy(s[e.name]),r=i[t];r!=null&&(r.defaultOutput=n,o.push(r))});let c=this.mapArgsToSignature(e);return{nodes:i,inputs:a,outputs:o,weights:r,placeholders:n,signature:c}}mapArgsToSignature(e){return{methodName:e.signature.name,inputs:e.signature.inputArg.reduce((e,t)=>(e[t.name]=this.mapArgToTensorInfo(t),e),{}),outputs:e.signature.outputArg.reduce((t,n)=>(t[n.name]=this.mapArgToTensorInfo(n,e.ret),t),{})}}mapArgToTensorInfo(e,t){let n=e.name;return t!=null&&(n=t[n]),{name:n,dtype:e.type}}};function mb(e){let t=O().global;if(t.atob!==void 0)return t.atob(e);if(typeof Buffer<`u`)return new Buffer(e,`base64`).toString();throw Error(`Unable to decode base64 in this environment. Missing built-in atob() or Buffer()`)}function hb(e,t){let n=Array.isArray(e)?String.fromCharCode.apply(null,e):mb(e);return t?n:n.toLowerCase()}function gb(e,t,n,r=!1){let i=e[t];return i==null?n:hb(i.s,r)}function _b(e,t,n){let r=e[t];return r?r.b:n}function vb(e,t,n){let r=e[t]||{},i=r.i==null?r.f==null?n:r.f:r.i;return typeof i==`number`?i:parseInt(i,10)}function yb(e){switch(typeof e==`string`&&(e=vy[e]),e){case vy.DT_FLOAT:case vy.DT_HALF:return`float32`;case vy.DT_INT32:case vy.DT_INT64:case vy.DT_INT8:case vy.DT_UINT8:return`int32`;case vy.DT_BOOL:return`bool`;case vy.DT_DOUBLE:return`float32`;case vy.DT_STRING:return`string`;case vy.DT_COMPLEX64:case vy.DT_COMPLEX128:return`complex64`;default:return null}}function bb(e,t,n){let r=e[t];return r&&r.func?r.func.name:n}function xb(e,t,n){let r=e[t];return r&&r.type?yb(r.type):n}function Sb(e,t,n){let r=e[t];return r&&r.list&&r.list.type?r.list.type.map(e=>yb(e)):n}function Cb(e){if(!e.unknownRank)return e.dim==null?[]:e.dim.map(e=>typeof e.size==`number`?e.size:parseInt(e.size,10))}function wb(e,t,n){let r=e[t];return r&&r.shape?Cb(r.shape):n}function Tb(e,t,n){let r=e[t];return r?((r.list.f&&r.list.f.length?r.list.f:r.list.i)||[]).map(e=>typeof e==`number`?e:parseInt(e,10)):n}function Eb(e,t,n,r=!1){let i=e[t];return i&&i.list&&i.list.s?i.list.s.map(e=>hb(e,r)):n}function Db(e,t,n){let r=e[t];return r&&r.list&&r.list.shape?r.list.shape.map(e=>Cb(e)):n}function Ob(e,t,n){let r=e[t];return r&&r.list&&r.list.b?r.list.b:n}var kb=class{constructor(e,t,n){this.node=e,this.tensorMap=t,this.context=n,this.inputs=[],this.attrs={},this.inputs=e.inputNames.map(e=>this.getInput(e)),e.rawAttrs!=null&&(this.attrs=Object.keys(e.rawAttrs).reduce((e,t)=>(e[t]=this.getAttr(t),e),{}))}getInput(e){return Sy(e,this.tensorMap,this.context)}getAttr(e,t){let n=this.node.rawAttrs[e];if(n.tensor!=null)return Sy(e,this.tensorMap,this.context);if(n.i!=null||n.f!=null)return vb(this.node.rawAttrs,e,t);if(n.s!=null)return gb(this.node.rawAttrs,e,t);if(n.b!=null)return _b(this.node.rawAttrs,e,t);if(n.shape!=null)return wb(this.node.rawAttrs,e,t);if(n.type!=null)return xb(this.node.rawAttrs,e,t);if(n.list!=null){if(n.list.i!=null||n.list.f!=null)return Tb(this.node.rawAttrs,e,t);if(n.list.s!=null)return Eb(this.node.rawAttrs,e,t);if(n.list.shape!=null)return Db(this.node.rawAttrs,e,t);if(n.list.b!=null)return Ob(this.node.rawAttrs,e,t);if(n.list.type!=null)return Sb(this.node.rawAttrs,e,t)}return t}},Ab=s({OP_SCOPE_SUFFIX:()=>ia,abs:()=>Ho,acos:()=>Wo,acosh:()=>Ko,add:()=>R,addN:()=>Jo,all:()=>Xo,any:()=>Qo,argMax:()=>es,argMin:()=>ns,asin:()=>is,asinh:()=>os,atan:()=>cs,atan2:()=>us,atanh:()=>fs,avgPool:()=>Ps,avgPool3d:()=>Is,basicLSTMCell:()=>Ks,batchNorm:()=>Zs,batchNorm2d:()=>$s,batchNorm3d:()=>tc,batchNorm4d:()=>rc,batchToSpaceND:()=>Js,bincount:()=>ac,bitwiseAnd:()=>sc,booleanMaskAsync:()=>vm,broadcastArgs:()=>lc,broadcastTo:()=>dc,buffer:()=>I,cast:()=>L,ceil:()=>pc,clipByValue:()=>gc,clone:()=>Po,complex:()=>oa,concat:()=>Rs,concat1d:()=>vc,concat2d:()=>bc,concat3d:()=>Sc,concat4d:()=>wc,conv1d:()=>Oc,conv2d:()=>Ec,conv2dTranspose:()=>Mc,conv3d:()=>Pc,conv3dTranspose:()=>Rc,cos:()=>Bc,cosh:()=>Hc,cosineWindow:()=>Pm,cumprod:()=>Wc,cumsum:()=>Kc,denseBincount:()=>Jc,depthToSpace:()=>Xc,depthwiseConv2d:()=>Qc,diag:()=>el,dilation2d:()=>nl,div:()=>z,divNoNan:()=>fl,dot:()=>ml,dropout:()=>Mm,einsum:()=>gl,elu:()=>vl,enclosingPowerOfTwo:()=>Nm,ensureShape:()=>bl,equal:()=>ol,erf:()=>Sl,euclideanNorm:()=>Kl,exp:()=>Jl,expandDims:()=>Xl,expm1:()=>Ql,eye:()=>nu,fft:()=>wp,fill:()=>mc,floor:()=>iu,floorDiv:()=>Ro,fused:()=>$m,gather:()=>ou,gatherND:()=>km,greater:()=>cu,greaterEqual:()=>uu,ifft:()=>Ep,imag:()=>fu,image:()=>Ug,inTopKAsync:()=>Im,irfft:()=>Op,isFinite:()=>mu,isInf:()=>gu,isNaN:()=>vu,leakyRelu:()=>bu,less:()=>Su,lessEqual:()=>wu,linalg:()=>Wg,linspace:()=>Tu,localResponseNormalization:()=>Du,log:()=>ku,log1p:()=>ju,logSigmoid:()=>zu,logSoftmax:()=>Hu,logSumExp:()=>Wu,logicalAnd:()=>Ku,logicalNot:()=>Ju,logicalOr:()=>Xu,logicalXor:()=>Qu,losses:()=>Gg,lowerBound:()=>nd,matMul:()=>H,max:()=>Ml,maxPool:()=>id,maxPool3d:()=>od,maxPoolWithArgmax:()=>cd,maximum:()=>ud,mean:()=>fd,meshgrid:()=>hd,min:()=>Pl,minimum:()=>_d,mirrorPad:()=>yd,mod:()=>xd,moments:()=>Cd,movingAverage:()=>Sm,mul:()=>B,multiRNNCell:()=>Td,multinomial:()=>Dd,neg:()=>Fu,norm:()=>Wl,notEqual:()=>kd,oneHot:()=>jd,ones:()=>md,onesLike:()=>Nd,op:()=>N,outerProduct:()=>Fd,pad:()=>Ld,pad1d:()=>zd,pad2d:()=>Vd,pad3d:()=>Ud,pad4d:()=>Gd,pool:()=>Zd,pow:()=>Il,prelu:()=>$d,print:()=>Fo,prod:()=>tf,raggedGather:()=>rf,raggedRange:()=>of,raggedTensorToTensor:()=>cf,rand:()=>uf,randomGamma:()=>Cf,randomNormal:()=>Tf,randomStandardNormal:()=>Df,randomUniform:()=>kf,randomUniformInt:()=>jf,range:()=>Mf,real:()=>Pf,reciprocal:()=>If,relu:()=>Rf,relu6:()=>Bf,reshape:()=>V,reverse:()=>Hf,reverse1d:()=>Wf,reverse2d:()=>Kf,reverse3d:()=>Jf,reverse4d:()=>Xf,rfft:()=>Mp,round:()=>Qf,rsqrt:()=>ep,scalar:()=>G,scatterND:()=>wm,searchSorted:()=>td,selu:()=>np,separableConv2d:()=>ip,setdiff1dAsync:()=>op,sigmoid:()=>Vs,sign:()=>cp,signal:()=>Hg,sin:()=>up,sinh:()=>fp,slice:()=>U,slice1d:()=>mp,slice2d:()=>gp,slice3d:()=>vp,slice4d:()=>bp,softmax:()=>Sp,softplus:()=>Lu,spaceToBatchND:()=>qd,sparse:()=>Kg,sparseToDense:()=>Dm,spectral:()=>Vg,split:()=>Ap,sqrt:()=>Rl,square:()=>Bl,squaredDifference:()=>Pp,squeeze:()=>Ip,stack:()=>Rp,step:()=>Bp,stridedSlice:()=>Hp,string:()=>qg,sub:()=>q,sum:()=>K,tan:()=>Wp,tanh:()=>Ws,tensor:()=>ca,tensor1d:()=>Gp,tensor2d:()=>Kp,tensor3d:()=>qp,tensor4d:()=>Jp,tensor5d:()=>Yp,tensor6d:()=>Xp,tensorScatterUpdate:()=>tm,tile:()=>eu,topk:()=>rm,transpose:()=>bm,truncatedNormal:()=>am,unique:()=>sm,unsortedSegmentSum:()=>lm,unstack:()=>dm,upperBound:()=>fm,variable:()=>pm,where:()=>cl,whereAsync:()=>gm,zeros:()=>pd,zerosLike:()=>ul}),jb=(e,t,n,r=Ab)=>{switch(e.op){case`BiasAdd`:case`AddV2`:case`Add`:return[r.add(J(`a`,e,t,n),J(`b`,e,t,n))];case`AddN`:return[r.addN(J(`tensors`,e,t,n))];case`FloorMod`:case`Mod`:return[r.mod(J(`a`,e,t,n),J(`b`,e,t,n))];case`Mul`:return[r.mul(J(`a`,e,t,n),J(`b`,e,t,n))];case`RealDiv`:case`Div`:return[r.div(J(`a`,e,t,n),J(`b`,e,t,n))];case`DivNoNan`:return[r.divNoNan(J(`a`,e,t,n),J(`b`,e,t,n))];case`FloorDiv`:return[r.floorDiv(J(`a`,e,t,n),J(`b`,e,t,n))];case`Sub`:return[r.sub(J(`a`,e,t,n),J(`b`,e,t,n))];case`Minimum`:return[r.minimum(J(`a`,e,t,n),J(`b`,e,t,n))];case`Maximum`:return[r.maximum(J(`a`,e,t,n),J(`b`,e,t,n))];case`Pow`:return[r.pow(J(`a`,e,t,n),J(`b`,e,t,n))];case`SquaredDifference`:return[r.squaredDifference(J(`a`,e,t,n),J(`b`,e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}},Mb=(e,t,n,r=Ab)=>{switch(e.op){case`Abs`:case`ComplexAbs`:return[r.abs(J(`x`,e,t,n))];case`Acos`:return[r.acos(J(`x`,e,t,n))];case`Acosh`:return[r.acosh(J(`x`,e,t,n))];case`Asin`:return[r.asin(J(`x`,e,t,n))];case`Asinh`:return[r.asinh(J(`x`,e,t,n))];case`Atan`:return[r.atan(J(`x`,e,t,n))];case`Atan2`:return[r.atan2(J(`x`,e,t,n),J(`y`,e,t,n))];case`Atanh`:return[r.atanh(J(`x`,e,t,n))];case`Ceil`:return[r.ceil(J(`x`,e,t,n))];case`Complex`:return[r.complex(J(`real`,e,t,n),J(`imag`,e,t,n))];case`Cos`:return[r.cos(J(`x`,e,t,n))];case`Cosh`:return[r.cosh(J(`x`,e,t,n))];case`Elu`:return[r.elu(J(`x`,e,t,n))];case`Erf`:return[r.erf(J(`x`,e,t,n))];case`Exp`:return[r.exp(J(`x`,e,t,n))];case`Expm1`:return[r.expm1(J(`x`,e,t,n))];case`Floor`:return[r.floor(J(`x`,e,t,n))];case`Log`:return[r.log(J(`x`,e,t,n))];case`Log1p`:return[r.log1p(J(`x`,e,t,n))];case`Imag`:return[r.imag(J(`x`,e,t,n))];case`Neg`:return[r.neg(J(`x`,e,t,n))];case`Reciprocal`:return[r.reciprocal(J(`x`,e,t,n))];case`Real`:return[r.real(J(`x`,e,t,n))];case`Relu`:return[r.relu(J(`x`,e,t,n))];case`Round`:return[r.round(J(`x`,e,t,n))];case`Selu`:return[r.selu(J(`x`,e,t,n))];case`Sigmoid`:return[r.sigmoid(J(`x`,e,t,n))];case`Sin`:return[r.sin(J(`x`,e,t,n))];case`Sign`:return[r.sign(J(`x`,e,t,n))];case`Sinh`:return[r.sinh(J(`x`,e,t,n))];case`Softplus`:return[r.softplus(J(`x`,e,t,n))];case`Sqrt`:return[r.sqrt(J(`x`,e,t,n))];case`Square`:return[r.square(J(`x`,e,t,n))];case`Tanh`:return[r.tanh(J(`x`,e,t,n))];case`Tan`:return[r.tan(J(`x`,e,t,n))];case`ClipByValue`:return[r.clipByValue(J(`x`,e,t,n),J(`clipValueMin`,e,t,n),J(`clipValueMax`,e,t,n))];case`Relu6`:return[r.relu6(J(`x`,e,t,n))];case`Rsqrt`:return[r.rsqrt(Sy(e.inputNames[0],t,n))];case`LeakyRelu`:return[r.leakyRelu(J(`x`,e,t,n),J(`alpha`,e,t,n))];case`Prelu`:return[r.prelu(J(`x`,e,t,n),J(`alpha`,e,t,n))];case`IsNan`:return[r.isNaN(Sy(e.inputNames[0],t,n))];case`IsInf`:return[r.isInf(Sy(e.inputNames[0],t,n))];case`IsFinite`:return[r.isFinite(Sy(e.inputNames[0],t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};function Nb(e,t,n=``){if(!(typeof e==`number`||typeof t==`number`)){y(e.length===t.length,()=>n+` Shapes ${e} and ${t} must match`);for(let r=0;r<e.length;r++){let i=e[r],a=t[r];y(i<0||a<0||i===a,()=>n+` Shapes ${e} and ${t} must match`)}}}function Pb(e){return!(typeof e==`number`||e.some(e=>e<0))}function Fb(e,t,n){let r=Ib(e,n),i=!Pb(r);if(i&&t.length===0)throw Error(`Tried to calculate elements of an empty list with non-fully-defined elementShape: ${r}`);if(i&&t.forEach(e=>{r=Ib(e.shape,r)}),!Pb(r))throw Error(`Non-fully-defined elementShape: ${r}`);return r}function Ib(e,t){if(typeof e==`number`)return t;if(typeof t==`number`)return e;if(e.length!==t.length)throw Error(`Incompatible ranks during merge: ${e} vs. ${t}`);let n=[];for(let r=0;r<e.length;++r){let i=e[r],a=t[r];if(i>=0&&a>=0&&i!==a)throw Error(`Incompatible shape during merge: ${e} vs. ${t}`);n[r]=i>=0?i:a}return n}var Lb=class{constructor(e,t,n,r,i,a,o){this.name=e,this.dtype=t,this.maxSize=n,this.elementShape=r,this.identicalElementShapes=i,this.dynamicSize=a,this.clearAfterRead=o,this.tensors=[],this.closed_=!1,this.idTensor=G(0),pa(this.idTensor)}get id(){return this.idTensor.id}get closed(){return this.closed_}clearAndClose(e){this.tensors.forEach(t=>{(e==null||!e.has(t.tensor.id))&&t.tensor.dispose()}),this.tensors=[],this.closed_=!0,this.idTensor.dispose()}size(){return this.tensors.length}read(e){if(this.closed_)throw Error(`TensorArray ${this.name} has already been closed.`);if(e<0||e>=this.size())throw Error(`Tried to read from index ${e}, but array size is: ${this.size()}`);let t=this.tensors[e];if(t.cleared)throw Error(`TensorArray ${this.name}: Could not read index ${e} twice because it was cleared after a previous read (perhaps try setting clear_after_read = false?).`);return this.clearAfterRead&&(t.cleared=!0),t.read=!0,t.tensor}readMany(e){return e.map(e=>this.read(e))}write(e,t){if(this.closed_)throw Error(`TensorArray ${this.name} has already been closed.`);if(e<0||!this.dynamicSize&&e>=this.maxSize)throw Error(`Tried to write to index ${e}, but array is not resizeable and size is: ${this.maxSize}`);let n=this.tensors[e]||{};if(t.dtype!==this.dtype)throw Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e},
          because the value dtype is ${t.dtype}, but TensorArray dtype is ${this.dtype}.`);if(this.size()===0&&(this.elementShape==null||this.elementShape.length===0)&&(this.elementShape=t.shape),Nb(this.elementShape,t.shape,`TensorArray ${this.name}: Could not write to TensorArray index ${e}.`),n.read)throw Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been read.`);if(n.written)throw Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been written.`);n.tensor=t,pa(t),n.written=!0,this.tensors[e]=n}writeMany(e,t){if(e.length!==t.length)throw Error(`TensorArray ${this.name}: could not write multiple tensors,because the index size: ${e.length} is not the same as tensors size: ${t.length}.`);e.forEach((e,n)=>this.write(e,t[n]))}gather(e,t){if(t&&t!==this.dtype)throw Error(`TensorArray dtype is ${this.dtype} but gather requested dtype ${t}`);if(e)e=e.slice(0,this.size());else{e=[];for(let t=0;t<this.size();t++)e.push(t)}if(e.length===0)return ca([],[0].concat(this.elementShape));let n=this.readMany(e);return Nb(this.elementShape,n[0].shape,`TensorArray shape mismatch: `),Rp(n,0)}concat(e){if(e&&e!==this.dtype)throw Error(`TensorArray dtype is ${this.dtype} but concat requested dtype ${e}`);if(this.size()===0)return ca([],[0].concat(this.elementShape));let t=[];for(let e=0;e<this.size();e++)t.push(e);let n=this.readMany(t);return Nb(this.elementShape,n[0].shape,`TensorArray shape mismatch: tensor array shape (${this.elementShape}) vs first tensor shape (${n[0].shape})`),Rs(n,0)}scatter(e,t){if(t.dtype!==this.dtype)throw Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${t.dtype}`);if(e.length!==t.shape[0])throw Error(`Expected len(indices) == tensor.shape[0], but saw: ${e.length} vs. ${t.shape[0]}`);let n=Math.max(...e);if(!this.dynamicSize&&n>=this.maxSize)throw Error(`Max index must be < array size (${n}  vs. ${this.maxSize})`);this.writeMany(e,dm(t,0))}split(e,t){if(t.dtype!==this.dtype)throw Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${t.dtype}`);let n=0,r=e.map(e=>(n+=e,n));if(n!==t.shape[0])throw Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${n}, and tensor's shape is: ${t.shape}`);if(!this.dynamicSize&&e.length!==this.maxSize)throw Error(`TensorArray's size is not equal to the size of lengths (${this.maxSize} vs. ${e.length}), and the TensorArray is not marked as dynamically resizeable`);let i=n===0?0:t.size/n,a=[];P(()=>{t=V(t,[1,n,i]);for(let n=0;n<e.length;++n){let o=[0,n===0?0:r[n-1],0],s=[1,e[n],i];a[n]=V(U(t,o,s),this.elementShape)}return a});let o=[];for(let t=0;t<e.length;t++)o[t]=t;this.writeMany(o,a)}},Rb=class e{get id(){return this.idTensor.id}constructor(e,t,n,r=-1){this.tensors=e,this.elementShape=t,this.elementDtype=n,e?.forEach(e=>{if(n!==e.dtype)throw Error(`Invalid data types; op elements ${n}, but list elements ${e.dtype}`);Nb(t,e.shape,`TensorList shape mismatch: `),pa(e)}),this.idTensor=G(0),this.maxNumElements=r,pa(this.idTensor)}copy(){return new e([...this.tensors],this.elementShape,this.elementDtype)}clearAndClose(e){this.tensors.forEach(t=>{(e==null||!e.has(t.id))&&t.dispose()}),this.tensors.length=0,this.idTensor.dispose()}size(){return this.tensors.length}stack(e,t,n=-1){if(t!==this.elementDtype)throw Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);if(n!==-1&&this.tensors.length!==n)throw Error(`Operation expected a list with ${n} elements but got a list with ${this.tensors.length} elements.`);Nb(e,this.elementShape,`TensorList shape mismatch: `);let r=Fb(this.elementShape,this.tensors,e);return P(()=>Rp(this.tensors.map(e=>V(e,r)),0))}popBack(e,t){if(t!==this.elementDtype)throw Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);if(this.size()===0)throw Error(`Trying to pop from an empty list.`);let n=Fb(this.elementShape,this.tensors,e),r=this.tensors.pop();return r.kept=!1,Nb(r.shape,e,`TensorList shape mismatch: `),V(r,n)}pushBack(e){if(e.dtype!==this.elementDtype)throw Error(`Invalid data types; op elements ${e.dtype}, but list elements ${this.elementDtype}`);if(Nb(e.shape,this.elementShape,`TensorList shape mismatch: `),this.maxNumElements===this.size())throw Error(`Trying to push element into a full list.`);pa(e),this.tensors.push(e)}resize(t){if(t<0)throw Error(`TensorListResize expects size to be non-negative. Got: ${t}`);if(this.maxNumElements!==-1&&t>this.maxNumElements)throw Error(`TensorListResize input size ${t} is greater maxNumElement ${this.maxNumElements}.`);let n=new e([],this.elementShape,this.elementDtype,this.maxNumElements);n.tensors.length=t;for(let e=0;e<Math.min(this.tensors.length,t);++e)n.tensors[e]=this.tensors[e];return n}getItem(e,t,n){if(n!==this.elementDtype)throw Error(`Invalid data types; op elements ${n}, but list elements ${this.elementDtype}`);if(e<0||e>this.tensors.length)throw Error(`Trying to access element ${e} in a list with ${this.tensors.length} elements.`);if(this.tensors[e]==null)throw Error(`element at index ${e} is null.`);Nb(this.tensors[e].shape,t,`TensorList shape mismatch: `);let r=Fb(this.elementShape,this.tensors,t);return V(this.tensors[e],r)}setItem(e,t){if(t.dtype!==this.elementDtype)throw Error(`Invalid data types; op elements ${t.dtype}, but list elements ${this.elementDtype}`);if(e<0||this.maxNumElements!==-1&&e>=this.maxNumElements)throw Error(`Trying to set element ${e} in a list with max ${this.maxNumElements} elements.`);Nb(this.elementShape,t.shape,`TensorList shape mismatch: `),pa(t),this.tensors[e]!=null&&(this.tensors[e].kept=!1),this.tensors[e]=t}gather(e,t,n){if(t!==this.elementDtype)throw Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);Nb(this.elementShape,n,`TensorList shape mismatch: `),e=e.slice(0,this.size());let r=Fb(this.elementShape,this.tensors,n);return e.length===0?ca([],[0].concat(r)):P(()=>Rp(e.map(e=>V(this.tensors[e],r)),0))}concat(e,t){if(e&&e!==this.elementDtype)throw Error(`TensorList dtype is ${this.elementDtype} but concat requested dtype ${e}`);Nb(this.elementShape,t,`TensorList shape mismatch: `);let n=Fb(this.elementShape,this.tensors,t);return this.size()===0?ca([],[0].concat(n)):P(()=>Rs(this.tensors.map(e=>V(e,n)),0))}};function zb(e,t,n){let r=e.dtype;if(e.shape.length<1)throw Error(`Tensor must be at least a vector, but saw shape: ${e.shape}`);if(e.dtype!==n)throw Error(`Invalid data types; op elements ${e.dtype}, but list elements ${n}`);return Nb(e.shape.slice(1),t,`TensorList shape mismatch: `),new Rb(dm(e),t,r)}function Bb(e,t,n,r){return new Rb([],e,t,r)}function Vb(e,t,n,r){if(t.length!==e.shape[0])throw Error(`Expected len(indices) == tensor.shape[0], but saw: ${t.length} vs. ${e.shape[0]}`);let i=Math.max(...t);if(r!=null&&r!==-1&&i>=r)throw Error(`Max index must be < array size (${i}  vs. ${r})`);let a=new Rb([],n,e.dtype,r),o=dm(e,0);return t.forEach((e,t)=>{a.setItem(e,o[t])}),a}function Hb(e,t,n){let r=0,i=t.map(e=>(r+=e,r));if(r!==e.shape[0])throw Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${r}, and tensor's shape is: ${e.shape}`);let a=Ib(e.shape.slice(1),n),o=r===0?0:e.size/r,s=P(()=>{let n=[];e=V(e,[1,r,o]);for(let r=0;r<t.length;++r){let s=[0,r===0?0:i[r-1],0],c=[1,t[r],o];n[r]=V(U(e,s,c),a)}return e.dispose(),n}),c=new Rb([],n,e.dtype,t.length);for(let e=0;e<s.length;e++)c.setItem(e,s[e]);return c}var Ub=async(e,t,n)=>{switch(e.op){case`If`:case`StatelessIf`:{let r=J(`thenBranch`,e,t,n),i=J(`elseBranch`,e,t,n),a=J(`cond`,e,t,n),o=J(`args`,e,t,n);return(await a.data())[0]?n.functionMap[r].executeFunctionAsync(o,n.tensorArrayMap,n.tensorListMap):n.functionMap[i].executeFunctionAsync(o,n.tensorArrayMap,n.tensorListMap)}case`While`:case`StatelessWhile`:{let r=J(`body`,e,t,n),i=J(`cond`,e,t,n),a=J(`args`,e,t,n),o=await n.functionMap[i].executeFunctionAsync(a,n.tensorArrayMap,n.tensorListMap),s=a.map(e=>e.id),c=await o[0].data();o.forEach(e=>{!e.kept&&s.indexOf(e.id)===-1&&e.dispose()});let l=a;for(;c[0];){let e=l;l=await n.functionMap[r].executeFunctionAsync(l,n.tensorArrayMap,n.tensorListMap);let t=l.map(e=>e.id);e.forEach(e=>{!e.kept&&s.indexOf(e.id)===-1&&t.indexOf(e.id)===-1&&e.dispose()});let a=await n.functionMap[i].executeFunctionAsync(l,n.tensorArrayMap,n.tensorListMap);c=await a[0].data(),a.forEach(e=>{!e.kept&&s.indexOf(e.id)===-1&&t.indexOf(e.id)===-1&&e.dispose()})}return l}case`LoopCond`:return[Oy(J(`pred`,e,t,n))];case`Switch`:{let r=J(`pred`,e,t,n),i=J(`data`,e,t,n);return i.kept||(i=Oy(i)),(await r.data())[0]?[void 0,i]:[i,void 0]}case`Merge`:{let r=e.inputNames.find(e=>Sy(e,t,n)!==void 0);return r?[Oy(Sy(r,t,n))]:void 0}case`Enter`:{let r=J(`frameName`,e,t,n),i=J(`tensor`,e,t,n);return n.enterFrame(r),[Oy(i)]}case`Exit`:{let r=J(`tensor`,e,t,n);return n.exitFrame(),[Oy(r)]}case`NextIteration`:{let r=J(`tensor`,e,t,n);return n.nextIteration(),[Oy(r)]}case`TensorArrayV3`:{let r=J(`size`,e,t,n),i=J(`dtype`,e,t,n),a=J(`elementShape`,e,t,n),o=J(`dynamicSize`,e,t,n),s=J(`clearAfterRead`,e,t,n),c=J(`identicalElementShapes`,e,t,n),l=new Lb(J(`name`,e,t,n),i,r,a,c,o,s);return n.addTensorArray(l),[l.idTensor,G(1)]}case`TensorArrayWriteV3`:{let r=J(`tensorArrayId`,e,t,n),i=J(`index`,e,t,n),a=J(`tensor`,e,t,n),o=n.getTensorArray(r.id);return o.write(i,a),[o.idTensor]}case`TensorArrayReadV3`:{let r=J(`tensorArrayId`,e,t,n),i=J(`index`,e,t,n);return[n.getTensorArray(r.id).read(i)]}case`TensorArrayGatherV3`:{let r=J(`tensorArrayId`,e,t,n),i=J(`indices`,e,t,n),a=J(`dtype`,e,t,n);return[n.getTensorArray(r.id).gather(i,a)]}case`TensorArrayScatterV3`:{let r=J(`tensorArrayId`,e,t,n),i=J(`indices`,e,t,n),a=J(`tensor`,e,t,n),o=n.getTensorArray(r.id);return o.scatter(i,a),[o.idTensor]}case`TensorArrayConcatV3`:{let r=J(`tensorArrayId`,e,t,n),i=n.getTensorArray(r.id),a=J(`dtype`,e,t,n);return[i.concat(a)]}case`TensorArraySplitV3`:{let r=J(`tensorArrayId`,e,t,n),i=J(`tensor`,e,t,n),a=J(`lengths`,e,t,n),o=n.getTensorArray(r.id);return o.split(a,i),[o.idTensor]}case`TensorArraySizeV3`:{let r=J(`tensorArrayId`,e,t,n);return[G(n.getTensorArray(r.id).size(),`int32`)]}case`TensorArrayCloseV3`:{let r=J(`tensorArrayId`,e,t,n),i=n.getTensorArray(r.id);return i.clearAndClose(),[i.idTensor]}case`TensorListSetItem`:{let r=J(`tensorListId`,e,t,n),i=J(`index`,e,t,n),a=J(`tensor`,e,t,n),o=n.getTensorList(r.id);return o.setItem(i,a),[o.idTensor]}case`TensorListGetItem`:{let r=J(`tensorListId`,e,t,n),i=J(`index`,e,t,n),a=J(`elementShape`,e,t,n),o=J(`elementDType`,e,t,n);return[n.getTensorList(r.id).getItem(i,a,o)]}case`TensorListScatterV2`:case`TensorListScatter`:{let r=J(`indices`,e,t,n),i=Vb(J(`tensor`,e,t,n),r,J(`elementShape`,e,t,n),J(`numElements`,e,t,n));return n.addTensorList(i),[i.idTensor]}case`TensorListReserve`:case`EmptyTensorList`:{let r=J(`elementShape`,e,t,n),i=J(`elementDType`,e,t,n),a;a=e.op===`TensorListReserve`?`numElements`:`maxNumElements`;let o=J(a,e,t,n),s=Bb(r,i,o,e.op===`TensorListReserve`?-1:o);return n.addTensorList(s),[s.idTensor]}case`TensorListGather`:{let r=J(`tensorListId`,e,t,n),i=J(`indices`,e,t,n),a=J(`elementShape`,e,t,n),o=J(`elementDType`,e,t,n);return[n.getTensorList(r.id).gather(i,o,a)]}case`TensorListStack`:{let r=J(`tensorListId`,e,t,n),i=J(`elementShape`,e,t,n),a=J(`elementDType`,e,t,n),o=J(`numElements`,e,t,n);return[n.getTensorList(r.id).stack(i,a,o)]}case`TensorListFromTensor`:{let r=zb(J(`tensor`,e,t,n),J(`elementShape`,e,t,n),J(`elementDType`,e,t,n));return n.addTensorList(r),[r.idTensor]}case`TensorListConcat`:case`TensorListConcatV2`:{let r=J(`tensorListId`,e,t,n),i=n.getTensorList(r.id),a=J(`dtype`,e,t,n),o=J(`elementShape`,e,t,n);return[i.concat(a,o)]}case`TensorListPushBack`:{let r=J(`tensorListId`,e,t,n),i=J(`tensor`,e,t,n),a=n.getTensorList(r.id);return a.pushBack(i),[a.idTensor]}case`TensorListPopBack`:{let r=J(`tensorListId`,e,t,n),i=J(`elementShape`,e,t,n),a=J(`elementDType`,e,t,n);return[n.getTensorList(r.id).popBack(i,a)]}case`TensorListSplit`:{let r=J(`tensor`,e,t,n),i=J(`elementShape`,e,t,n),a=Hb(r,J(`lengths`,e,t,n),i);return n.addTensorList(a),[a.idTensor]}case`TensorListLength`:{let r=J(`tensorListId`,e,t,n);return[G(n.getTensorList(r.id).size(),`int32`)]}case`TensorListResize`:{let r=J(`tensorListId`,e,t,n),i=J(`size`,e,t,n),a=n.getTensorList(r.id).resize(i);return n.addTensorList(a),[a.idTensor]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};function Wb(e,t,n){let[r,i]=J(`fusedOps`,e,t,n),a=r===`biasadd`,o=!a,s=i===`prelu`,c=r===`fusedbatchnorm`,l=J(`numArgs`,e,t,n);if(a){if(s&&l!==2)throw Error(`FusedConv2d and DepthwiseConv2d with BiasAdd and Prelu must have two extra arguments: bias and alpha.`);if(!s&&a&&l!==1)throw Error(`FusedConv2d and DepthwiseConv2d with BiasAdd must have one extra argument: bias.`)}if(c)throw Error(`FusedConv2d and DepthwiseConv2d with FusedBatchNorm is not supported`);let u=J(`strides`,e,t,n),d=Dy(e,t,n),f=J(`dataFormat`,e,t,n).toUpperCase(),p=J(`dilations`,e,t,n),[m,h]=J(`args`,e,t,n);o&&(h=m,m=void 0);let g=J(`leakyreluAlpha`,e,t,n);return{stride:u,pad:d,dataFormat:f,dilations:p,biasArg:m,preluArg:h,activationFunc:i,leakyreluAlpha:g}}var Gb=(e,t,n,r=Ab)=>{switch(e.op){case`Conv1D`:{let i=J(`stride`,e,t,n),a=J(`pad`,e,t,n),o=J(`dataFormat`,e,t,n).toUpperCase(),s=J(`dilation`,e,t,n);return[r.conv1d(J(`x`,e,t,n),J(`filter`,e,t,n),i,a,o,s)]}case`Conv2D`:{let i=J(`strides`,e,t,n),a=Dy(e,t,n),o=J(`dataFormat`,e,t,n).toUpperCase(),s=J(`dilations`,e,t,n);return[r.conv2d(J(`x`,e,t,n),J(`filter`,e,t,n),[i[1],i[2]],a,o,[s[1],s[2]])]}case`_FusedConv2D`:{let{stride:i,pad:a,dataFormat:o,dilations:s,biasArg:c,preluArg:l,activationFunc:u,leakyreluAlpha:d}=Wb(e,t,n);return[r.fused.conv2d({x:J(`x`,e,t,n),filter:J(`filter`,e,t,n),strides:[i[1],i[2]],pad:a,dataFormat:o,dilations:[s[1],s[2]],bias:c,activation:u,preluActivationWeights:l,leakyreluAlpha:d})]}case`FusedDepthwiseConv2dNative`:{let{stride:i,pad:a,dataFormat:o,dilations:s,biasArg:c,preluArg:l,activationFunc:u,leakyreluAlpha:d}=Wb(e,t,n);return[r.fused.depthwiseConv2d({x:J(`x`,e,t,n),filter:J(`filter`,e,t,n),strides:[i[1],i[2]],pad:a,dataFormat:o,dilations:[s[1],s[2]],bias:c,activation:u,preluActivationWeights:l,leakyreluAlpha:d})]}case`Conv2DBackpropInput`:case`Conv2dTranspose`:{let i=J(`outputShape`,e,t,n),a=J(`strides`,e,t,n),o=Dy(e,t,n);return[r.conv2dTranspose(J(`x`,e,t,n),J(`filter`,e,t,n),i,[a[1],a[2]],o)]}case`DepthwiseConv2dNative`:case`DepthwiseConv2d`:{let i=J(`strides`,e,t,n),a=Dy(e,t,n),o=J(`dilations`,e,t,n),s=J(`dataFormat`,e,t,n).toUpperCase();return[r.depthwiseConv2d(J(`input`,e,t,n),J(`filter`,e,t,n),[i[1],i[2]],a,s,[o[1],o[2]])]}case`Conv3D`:{let i=J(`strides`,e,t,n),a=J(`pad`,e,t,n),o=J(`dataFormat`,e,t,n).toUpperCase(),s=J(`dilations`,e,t,n);return[r.conv3d(J(`x`,e,t,n),J(`filter`,e,t,n),[i[1],i[2],i[3]],a,o,[s[1],s[2],s[3]])]}case`AvgPool`:{let i=J(`strides`,e,t,n),a=J(`pad`,e,t,n),o=J(`kernelSize`,e,t,n);return[r.avgPool(J(`x`,e,t,n),[o[1],o[2]],[i[1],i[2]],a)]}case`MaxPool`:{let i=J(`strides`,e,t,n),a=J(`pad`,e,t,n),o=J(`kernelSize`,e,t,n);return[r.maxPool(J(`x`,e,t,n),[o[1],o[2]],[i[1],i[2]],a)]}case`MaxPoolWithArgmax`:{let i=J(`strides`,e,t,n),a=J(`pad`,e,t,n),o=J(`kernelSize`,e,t,n),s=J(`includeBatchInIndex`,e,t,n),{result:c,indexes:l}=r.maxPoolWithArgmax(J(`x`,e,t,n),[o[1],o[2]],[i[1],i[2]],a,s);return[c,l]}case`AvgPool3D`:{let i=J(`strides`,e,t,n),a=J(`pad`,e,t,n),o=J(`kernelSize`,e,t,n);return[r.avgPool3d(J(`x`,e,t,n),[o[1],o[2],o[3]],[i[1],i[2],i[3]],a)]}case`MaxPool3D`:{let i=J(`strides`,e,t,n),a=J(`pad`,e,t,n),o=J(`kernelSize`,e,t,n);return[r.maxPool3d(J(`x`,e,t,n),[o[1],o[2],o[3]],[i[1],i[2],i[3]],a)]}case`Dilation2D`:{let i=J(`strides`,e,t,n),a=J(`pad`,e,t,n),o=J(`dilations`,e,t,n),s=i[1],c=i[2],l=o[1],u=o[2];return[r.dilation2d(J(`x`,e,t,n),J(`filter`,e,t,n),[s,c],a,[l,u],`NHWC`)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},Kb=(e,t,n,r=Ab)=>{switch(e.op){case`Fill`:{let i=J(`shape`,e,t,n),a=J(`dtype`,e,t,n),o=J(`value`,e,t,n);return[r.fill(i,o,a)]}case`LinSpace`:{let i=J(`start`,e,t,n),a=J(`stop`,e,t,n),o=J(`num`,e,t,n);return[r.linspace(i,a,o)]}case`Multinomial`:{let i=J(`logits`,e,t,n),a=J(`numSamples`,e,t,n),o=J(`seed`,e,t,n);return[r.multinomial(i,a,o)]}case`OneHot`:{let i=J(`indices`,e,t,n),a=J(`depth`,e,t,n),o=J(`onValue`,e,t,n),s=J(`offValue`,e,t,n),c=J(`dtype`,e,t,n);return[r.oneHot(i,a,o,s,c)]}case`Ones`:return[r.ones(J(`shape`,e,t,n),J(`dtype`,e,t,n))];case`OnesLike`:return[r.onesLike(J(`x`,e,t,n))];case`RandomStandardNormal`:return[r.randomStandardNormal(J(`shape`,e,t,n),J(`dtype`,e,t,n),J(`seed`,e,t,n))];case`RandomUniform`:return[r.randomUniform(J(`shape`,e,t,n),J(`minval`,e,t,n),J(`maxval`,e,t,n),J(`dtype`,e,t,n))];case`RandomUniformInt`:return[r.randomUniformInt(J(`shape`,e,t,n),J(`minval`,e,t,n),J(`maxval`,e,t,n),J(`seed`,e,t,n))];case`Range`:{let i=J(`start`,e,t,n),a=J(`stop`,e,t,n),o=J(`step`,e,t,n);return[r.range(i,a,o,J(`dtype`,e,t,n))]}case`TruncatedNormal`:{let i=J(`shape`,e,t,n),a=J(`mean`,e,t,n),o=J(`stdDev`,e,t,n),s=J(`seed`,e,t,n);return[r.truncatedNormal(i,a,o,J(`dtype`,e,t,n),s)]}case`Zeros`:return[r.zeros(J(`shape`,e,t,n),J(`dtype`,e,t,n))];case`ZerosLike`:return[r.zerosLike(J(`x`,e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};function qb(e,t,n){return{boxes:J(`boxes`,e,t,n),scores:J(`scores`,e,t,n),maxOutputSize:J(`maxOutputSize`,e,t,n),iouThreshold:J(`iouThreshold`,e,t,n),scoreThreshold:J(`scoreThreshold`,e,t,n),softNmsSigma:J(`softNmsSigma`,e,t,n)}}var Jb=async(e,t,n,r,i=Ab)=>{switch(e.op){case`NonMaxSuppressionV5`:{let{boxes:r,scores:a,maxOutputSize:o,iouThreshold:s,scoreThreshold:c,softNmsSigma:l}=qb(e,t,n),u=await i.image.nonMaxSuppressionWithScoreAsync(r,a,o,s,c,l);return[u.selectedIndices,u.selectedScores]}case`NonMaxSuppressionV4`:{let{boxes:r,scores:a,maxOutputSize:o,iouThreshold:s,scoreThreshold:c}=qb(e,t,n),l=J(`padToMaxOutputSize`,e,t,n),u=await i.image.nonMaxSuppressionPaddedAsync(r,a,o,s,c,l);return[u.selectedIndices,u.validOutputs]}case`NonMaxSuppressionV3`:case`NonMaxSuppressionV2`:{let{boxes:r,scores:a,maxOutputSize:o,iouThreshold:s,scoreThreshold:c}=qb(e,t,n);return[await i.image.nonMaxSuppressionAsync(r,a,o,s,c)]}case`Where`:{let r=i.cast(J(`condition`,e,t,n),`bool`),a=[await i.whereAsync(r)];return r.dispose(),a}case`ListDiff`:return i.setdiff1dAsync(J(`x`,e,t,n),J(`y`,e,t,n));default:throw TypeError(`Node type ${e.op} is not implemented`)}},Yb=(e,t,n,r=Ab)=>{switch(e.op){case`LowerBound`:{let i=J(`sortedSequence`,e,t,n),a=J(`values`,e,t,n);return[r.lowerBound(i,a)]}case`TopKV2`:{let i=J(`x`,e,t,n),a=J(`k`,e,t,n),o=J(`sorted`,e,t,n),s=r.topk(i,a,o);return[s.values,s.indices]}case`UpperBound`:{let i=J(`sortedSequence`,e,t,n),a=J(`values`,e,t,n);return[r.upperBound(i,a)]}case`Unique`:{let i=J(`x`,e,t,n),a=r.unique(i);return[a.values,a.indices]}case`UniqueV2`:{let i=J(`x`,e,t,n),a=J(`axis`,e,t,n),o=r.unique(i,a);return[o.values,o.indices]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},Xb=(e,t,n,r=Ab)=>{switch(e.op){case`Const`:return t[e.name];case`PlaceholderWithDefault`:let i=J(`default`,e,t,n);return[Sy(e.name,t,n)||i];case`Placeholder`:return[Sy(e.name,t,n)];case`Identity`:case`StopGradient`:case`FakeQuantWithMinMaxVars`:return[Oy(J(`x`,e,t,n))];case`IdentityN`:return J(`x`,e,t,n).map(e=>Oy(e));case`Snapshot`:return[Oy(J(`x`,e,t,n))];case`Shape`:return[r.tensor1d(J(`x`,e,t,n).shape,`int32`)];case`ShapeN`:return J(`x`,e,t,n).map(e=>r.tensor1d(e.shape));case`Size`:return[r.scalar(J(`x`,e,t,n).size,`int32`)];case`Rank`:return[r.scalar(J(`x`,e,t,n).rank,`int32`)];case`NoOp`:return[r.scalar(1)];case`Print`:let a=J(`x`,e,t,n),o=J(`data`,e,t,n),s=J(`message`,e,t,n),c=J(`summarize`,e,t,n);console.warn(`The graph has a tf.print() operation,usually used for debugging, which slows down performance.`),console.log(s);for(let e=0;e<o.length;e++)console.log(Array.prototype.slice.call(o[e].dataSync()).slice(0,c));return[a];default:throw TypeError(`Node type ${e.op} is not implemented`)}},Zb=class{get id(){return this.handle.id}constructor(e,t){this.keyDType=e,this.valueDType=t,this.handle=G(0),this.tensorMap=new Map,pa(this.handle)}clearAndClose(){this.tensorMap.forEach(e=>e.dispose()),this.tensorMap.clear(),this.handle.dispose()}size(){return this.tensorMap.size}tensorSize(){return G(this.size(),`int32`)}async import(e,t){this.checkKeyAndValueTensor(e,t);let n=await e.data();return this.tensorMap.forEach(e=>e.dispose()),this.tensorMap.clear(),P(()=>{let e=dm(t),r=n.length,i=e.length;y(r===i,()=>`The number of elements doesn't match, keys has ${r} elements, the values has ${i} elements.`);for(let t=0;t<r;t++){let r=n[t],i=e[t];pa(i),this.tensorMap.set(r,i)}return this.handle})}async find(e,t){this.checkKeyAndValueTensor(e,t);let n=await e.data();return P(()=>{let e=[];for(let r=0;r<n.length;r++){let i=n[r],a=this.findWithDefault(i,t);e.push(a)}return Rp(e)})}findWithDefault(e,t){return this.tensorMap.get(e)??t}checkKeyAndValueTensor(e,t){if(e.dtype!==this.keyDType)throw Error(`Expect key dtype ${this.keyDType}, but got ${e.dtype}`);if(t.dtype!==this.valueDType)throw Error(`Expect value dtype ${this.valueDType}, but got ${t.dtype}`)}},Qb=async(e,t,n,r)=>{switch(e.op){case`HashTable`:case`HashTableV2`:{let i=r.getHashTableHandleByName(e.name);if(i!=null)return[i];{let i=new Zb(J(`keyDType`,e,t,n),J(`valueDType`,e,t,n));return r.addHashTable(e.name,i),[i.handle]}}case`InitializeTable`:case`InitializeTableV2`:case`LookupTableImport`:case`LookupTableImportV2`:{let i=J(`tableHandle`,e,t,n,r),a=J(`keys`,e,t,n),o=J(`values`,e,t,n);return[await r.getHashTableById(i.id).import(a,o)]}case`LookupTableFind`:case`LookupTableFindV2`:{let i=J(`tableHandle`,e,t,n,r),a=J(`keys`,e,t,n),o=J(`defaultValue`,e,t,n);return[await r.getHashTableById(i.id).find(a,o)]}case`LookupTableSize`:case`LookupTableSizeV2`:{let i=J(`tableHandle`,e,t,n,r);return[r.getHashTableById(i.id).tensorSize()]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},$b=(e,t,n,r=Ab)=>{switch(e.op){case`ResizeBilinear`:{let i=J(`images`,e,t,n),a=J(`size`,e,t,n),o=J(`alignCorners`,e,t,n),s=J(`halfPixelCenters`,e,t,n);return[r.image.resizeBilinear(i,[a[0],a[1]],o,s)]}case`ResizeNearestNeighbor`:{let i=J(`images`,e,t,n),a=J(`size`,e,t,n),o=J(`alignCorners`,e,t,n),s=J(`halfPixelCenters`,e,t,n);return[r.image.resizeNearestNeighbor(i,[a[0],a[1]],o,s)]}case`CropAndResize`:{let i=J(`image`,e,t,n),a=J(`boxes`,e,t,n),o=J(`boxInd`,e,t,n),s=J(`cropSize`,e,t,n),c=J(`method`,e,t,n),l=J(`extrapolationValue`,e,t,n);return[r.image.cropAndResize(i,a,o,s,c,l)]}case`ImageProjectiveTransformV3`:{let i=J(`images`,e,t,n),a=J(`transforms`,e,t,n),o=J(`outputShape`,e,t,n),s=J(`fillValue`,e,t,n),c=J(`interpolation`,e,t,n),l=J(`fillMode`,e,t,n);return[r.image.transform(i,a,c.toLowerCase(),l.toLowerCase(),s,o)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},ex=(e,t,n,r=Ab)=>{switch(e.op){case`Equal`:return[r.equal(J(`a`,e,t,n),J(`b`,e,t,n))];case`NotEqual`:return[r.notEqual(J(`a`,e,t,n),J(`b`,e,t,n))];case`Greater`:return[r.greater(J(`a`,e,t,n),J(`b`,e,t,n))];case`GreaterEqual`:return[r.greaterEqual(J(`a`,e,t,n),J(`b`,e,t,n))];case`Less`:return[r.less(J(`a`,e,t,n),J(`b`,e,t,n))];case`LessEqual`:return[r.lessEqual(J(`a`,e,t,n),J(`b`,e,t,n))];case`LogicalAnd`:return[r.logicalAnd(J(`a`,e,t,n),J(`b`,e,t,n))];case`LogicalNot`:return[r.logicalNot(J(`a`,e,t,n))];case`LogicalOr`:return[r.logicalOr(J(`a`,e,t,n),J(`b`,e,t,n))];case`Select`:case`SelectV2`:return[r.where(J(`condition`,e,t,n),J(`a`,e,t,n),J(`b`,e,t,n))];case`BitwiseAnd`:return[r.bitwiseAnd(J(`a`,e,t,n),J(`b`,e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}},tx=(e,t,n,r=Ab)=>{switch(e.op){case`BatchMatMul`:case`BatchMatMulV2`:case`MatMul`:return[r.matMul(J(`a`,e,t,n),J(`b`,e,t,n),J(`transposeA`,e,t,n),J(`transposeB`,e,t,n))];case`Einsum`:return[r.einsum(J(`equation`,e,t,n),...J(`tensors`,e,t,n))];case`Transpose`:return[r.transpose(J(`x`,e,t,n),J(`perm`,e,t,n))];case`_FusedMatMul`:let[i,a]=J(`fusedOps`,e,t,n),o=i===`biasadd`,s=a===`prelu`,c=J(`numArgs`,e,t,n),l=J(`leakyreluAlpha`,e,t,n);if(o){if(s&&c!==2)throw Error(`Fused MatMul with BiasAdd and Prelu must have two extra arguments: bias and alpha.`);if(!s&&c!==1)throw Error(`Fused MatMul with BiasAdd must have one extra argument: bias.`)}let[u,d]=J(`args`,e,t,n);return[r.fused.matMul({a:J(`a`,e,t,n),b:J(`b`,e,t,n),transposeA:J(`transposeA`,e,t,n),transposeB:J(`transposeB`,e,t,n),bias:u,activation:a,preluActivationWeights:d,leakyreluAlpha:l})];case`MatrixBandPart`:return[r.linalg.bandPart(J(`a`,e,t,n),J(`numLower`,e,t,n),J(`numUpper`,e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}},nx=(e,t,n,r=Ab)=>{switch(e.op){case`EuclideanNorm`:return[r.euclideanNorm(J(`x`,e,t,n),J(`axis`,e,t,n),J(`keepDims`,e,t,n))];case`FusedBatchNorm`:case`FusedBatchNormV2`:return[r.batchNorm(J(`x`,e,t,n),J(`mean`,e,t,n),J(`variance`,e,t,n),J(`offset`,e,t,n),J(`scale`,e,t,n),J(`epsilon`,e,t,n))];case`FusedBatchNormV3`:return[r.batchNorm(J(`x`,e,t,n),J(`mean`,e,t,n),J(`variance`,e,t,n),J(`offset`,e,t,n),J(`scale`,e,t,n),J(`epsilon`,e,t,n))];case`LRN`:return[r.localResponseNormalization(J(`x`,e,t,n),J(`radius`,e,t,n),J(`bias`,e,t,n),J(`alpha`,e,t,n),J(`beta`,e,t,n))];case`Softmax`:return[r.softmax(J(`x`,e,t,n))];case`LogSoftmax`:return[r.logSoftmax(J(`x`,e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}},rx=(e,t,n,r=Ab)=>{switch(e.op){case`RaggedGather`:{let{outputNestedSplits:i,outputDenseValues:a}=r.raggedGather(J(`paramsNestedSplits`,e,t,n),J(`paramsDenseValues`,e,t,n),J(`indices`,e,t,n),J(`outputRaggedRank`,e,t,n));return i.concat(a)}case`RaggedRange`:{let{rtNestedSplits:i,rtDenseValues:a}=r.raggedRange(J(`starts`,e,t,n),J(`limits`,e,t,n),J(`splits`,e,t,n));return[i,a]}case`RaggedTensorToTensor`:return[r.raggedTensorToTensor(J(`shape`,e,t,n),J(`values`,e,t,n),J(`defaultValue`,e,t,n),J(`rowPartitionTensors`,e,t,n),J(`rowPartitionTypes`,e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}},ix=(e,t,n,r=Ab)=>{switch(e.op){case`Max`:{let i=J(`axis`,e,t,n),a=J(`keepDims`,e,t,n);return[r.max(J(`x`,e,t,n),i,a)]}case`Mean`:{let i=J(`axis`,e,t,n),a=J(`keepDims`,e,t,n);return[r.mean(J(`x`,e,t,n),i,a)]}case`Min`:{let i=J(`axis`,e,t,n),a=J(`keepDims`,e,t,n);return[r.min(J(`x`,e,t,n),i,a)]}case`Sum`:{let i=J(`axis`,e,t,n),a=J(`keepDims`,e,t,n);return[r.sum(J(`x`,e,t,n),i,a)]}case`All`:{let i=J(`axis`,e,t,n),a=J(`keepDims`,e,t,n);return[r.all(J(`x`,e,t,n),i,a)]}case`Any`:{let i=J(`axis`,e,t,n),a=J(`keepDims`,e,t,n);return[r.any(J(`x`,e,t,n),i,a)]}case`ArgMax`:{let i=J(`axis`,e,t,n);return[r.argMax(J(`x`,e,t,n),i)]}case`ArgMin`:{let i=J(`axis`,e,t,n);return[r.argMin(J(`x`,e,t,n),i)]}case`Prod`:{let i=J(`axis`,e,t,n),a=J(`keepDims`,e,t,n);return[r.prod(J(`x`,e,t,n),i,a)]}case`Cumprod`:{let i=J(`axis`,e,t,n),a=J(`exclusive`,e,t,n),o=J(`reverse`,e,t,n);return[r.cumprod(J(`x`,e,t,n),i,a,o)]}case`Cumsum`:{let i=J(`axis`,e,t,n),a=J(`exclusive`,e,t,n),o=J(`reverse`,e,t,n);return[r.cumsum(J(`x`,e,t,n),i,a,o)]}case`Bincount`:let i=J(`x`,e,t,n),a=J(`weights`,e,t,n),o=J(`size`,e,t,n);return[r.bincount(i,a,o)];case`DenseBincount`:{let i=J(`x`,e,t,n),a=J(`weights`,e,t,n),o=J(`size`,e,t,n),s=J(`binaryOutput`,e,t,n);return[r.denseBincount(i,a,o,s)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},ax=(e,t,n,r=Ab)=>{switch(e.op){case`ConcatV2`:case`Concat`:{let i=J(`n`,e,t,n),a=J(`axis`,e,t,n),o=J(`tensors`,e,t,n);return o=o.slice(0,i),[r.concat(o,a)]}case`Gather`:{let i=J(`x`,e,t,n),a=J(`indices`,e,t,n);return[r.gather(i,r.cast(a,`int32`),0)]}case`GatherV2`:{let i=J(`axis`,e,t,n),a=J(`batchDims`,e,t,n),o=J(`x`,e,t,n),s=J(`indices`,e,t,n);return[r.gather(o,r.cast(s,`int32`),i,a)]}case`Reverse`:{let i=J(`dims`,e,t,n),a=[];for(let e=0;e<i.length;e++)i[e]&&a.push(e);let o=J(`x`,e,t,n);return[r.reverse(o,a)]}case`ReverseV2`:{let i=J(`axis`,e,t,n),a=J(`x`,e,t,n);return[r.reverse(a,i)]}case`Slice`:{let i=J(`begin`,e,t,n),a=J(`size`,e,t,n);return[r.slice(J(`x`,e,t,n),i,a)]}case`StridedSlice`:{let i=J(`begin`,e,t,n),a=J(`end`,e,t,n),o=J(`strides`,e,t,n),s=J(`beginMask`,e,t,n),c=J(`endMask`,e,t,n),l=J(`ellipsisMask`,e,t,n),u=J(`newAxisMask`,e,t,n),d=J(`shrinkAxisMask`,e,t,n),f=J(`x`,e,t,n);return[r.stridedSlice(f,i,a,o,s,c,l,u,d)]}case`Pack`:return P(()=>{let i=J(`axis`,e,t,n),a=J(`tensors`,e,t,n),o=a[0].shape,s=r.squeeze(a[0]).shape,c=a.map(e=>{let t=w(e.shape,o);if(!t&&!w(r.squeeze(e).shape,s))throw Error(`the input tensors shape does not match`);return t?e:r.reshape(e,o)});return[r.stack(c,i)]});case`Unpack`:{let i=J(`axis`,e,t,n),a=J(`tensor`,e,t,n);return r.unstack(a,i)}case`Tile`:{let i=J(`reps`,e,t,n);return[r.tile(J(`x`,e,t,n),i)]}case`Split`:case`SplitV`:{let i=J(`axis`,e,t,n),a=J(`numOrSizeSplits`,e,t,n),o=J(`x`,e,t,n);return r.split(o,a,i)}case`ScatterNd`:{let i=J(`indices`,e,t,n),a=J(`values`,e,t,n),o=J(`shape`,e,t,n);return[r.scatterND(i,a,o)]}case`GatherNd`:{let i=J(`x`,e,t,n),a=J(`indices`,e,t,n);return[r.gatherND(i,a)]}case`SparseToDense`:{let i=J(`sparseIndices`,e,t,n),a=J(`outputShape`,e,t,n),o=J(`sparseValues`,e,t,n),s=J(`defaultValue`,e,t,n);return[r.sparseToDense(i,o,a,o.dtype===s.dtype?s:r.cast(s,o.dtype))]}case`TensorScatterUpdate`:{let i=J(`indices`,e,t,n),a=J(`values`,e,t,n),o=J(`tensor`,e,t,n);return[r.tensorScatterUpdate(o,i,a)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},ox=(e,t,n,r=Ab)=>{switch(e.op){case`SparseFillEmptyRows`:{let{outputIndices:i,outputValues:a,emptyRowIndicator:o,reverseIndexMap:s}=r.sparse.sparseFillEmptyRows(J(`indices`,e,t,n),J(`values`,e,t,n),J(`denseShape`,e,t,n),J(`defaultValue`,e,t,n));return[i,a,o,s]}case`SparseReshape`:{let{outputIndices:i,outputShape:a}=r.sparse.sparseReshape(J(`inputIndices`,e,t,n),J(`inputShape`,e,t,n),J(`newShape`,e,t,n));return[i,a]}case`SparseSegmentMean`:return[r.sparse.sparseSegmentMean(J(`data`,e,t,n),J(`indices`,e,t,n),J(`segmentIds`,e,t,n))];case`SparseSegmentSum`:return[r.sparse.sparseSegmentSum(J(`data`,e,t,n),J(`indices`,e,t,n),J(`segmentIds`,e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}},sx=(e,t,n,r=Ab)=>{switch(e.op){case`FFT`:return[r.fft(J(`x`,e,t,n))];case`IFFT`:return[r.ifft(J(`x`,e,t,n))];case`RFFT`:return[r.rfft(J(`x`,e,t,n))];case`IRFFT`:return[r.irfft(J(`x`,e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}},cx=(e,t,n,r=Ab)=>{switch(e.op){case`StaticRegexReplace`:return[r.string.staticRegexReplace(J(`input`,e,t,n),J(`pattern`,e,t,n),J(`rewrite`,e,t,n),J(`replaceGlobal`,e,t,n))];case`StringNGrams`:{let{nGrams:i,nGramsSplits:a}=r.string.stringNGrams(J(`data`,e,t,n),J(`dataSplits`,e,t,n),J(`separator`,e,t,n),J(`nGramWidths`,e,t,n),J(`leftPad`,e,t,n),J(`rightPad`,e,t,n),J(`padWidth`,e,t,n),J(`preserveShortSequences`,e,t,n));return[i,a]}case`StringSplit`:{let{indices:i,values:a,shape:o}=r.string.stringSplit(J(`input`,e,t,n),J(`delimiter`,e,t,n),J(`skipEmpty`,e,t,n));return[i,a,o]}case`StringToHashBucketFast`:return[r.string.stringToHashBucketFast(J(`input`,e,t,n),J(`numBuckets`,e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}},lx=(e,t,n,r=Ab)=>{switch(e.op){case`Cast`:return[r.cast(J(`x`,e,t,n),J(`dtype`,e,t,n))];case`ExpandDims`:{let i=J(`axis`,e,t,n);return[r.expandDims(J(`x`,e,t,n),i)]}case`Squeeze`:{let i=J(`axis`,e,t,n);return[r.squeeze(J(`x`,e,t,n),i)]}case`Reshape`:return[r.reshape(J(`x`,e,t,n),J(`shape`,e,t,n))];case`EnsureShape`:return[r.ensureShape(J(`x`,e,t,n),J(`shape`,e,t,n))];case`MirrorPad`:return[r.mirrorPad(J(`x`,e,t,n),J(`padding`,e,t,n),J(`mode`,e,t,n))];case`PadV2`:case`Pad`:return[r.pad(J(`x`,e,t,n),J(`padding`,e,t,n),J(`constantValue`,e,t,n))];case`SpaceToBatchND`:{let i=J(`blockShape`,e,t,n),a=J(`paddings`,e,t,n);return[r.spaceToBatchND(J(`x`,e,t,n),i,a)]}case`BatchToSpaceND`:{let i=J(`blockShape`,e,t,n),a=J(`crops`,e,t,n);return[r.batchToSpaceND(J(`x`,e,t,n),i,a)]}case`DepthToSpace`:{let i=J(`blockSize`,e,t,n),a=J(`dataFormat`,e,t,n).toUpperCase();return[r.depthToSpace(J(`x`,e,t,n),i,a)]}case`BroadcastTo`:return[r.broadcastTo(J(`x`,e,t,n),J(`shape`,e,t,n))];case`BroadcastArgs`:return[r.broadcastArgs(J(`s0`,e,t,n),J(`s1`,e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};function ux(e,t,n,r,i=P){let a=((e,t,n)=>{switch(e.category){case`arithmetic`:return i(()=>jb(e,t,n));case`basic_math`:return i(()=>Mb(e,t,n));case`control`:return Ub(e,t,n);case`convolution`:return i(()=>Gb(e,t,n));case`creation`:return i(()=>Kb(e,t,n));case`dynamic`:return Jb(e,t,n);case`evaluation`:return i(()=>Yb(e,t,n));case`image`:return i(()=>$b(e,t,n));case`graph`:return i(()=>Xb(e,t,n));case`logical`:return i(()=>ex(e,t,n));case`matrices`:return i(()=>tx(e,t,n));case`normalization`:return i(()=>nx(e,t,n));case`ragged`:return i(()=>rx(e,t,n));case`reduction`:return i(()=>ix(e,t,n));case`slice_join`:return i(()=>ax(e,t,n));case`sparse`:return i(()=>ox(e,t,n));case`spectral`:return i(()=>sx(e,t,n));case`string`:return i(()=>cx(e,t,n));case`transformation`:return i(()=>lx(e,t,n));case`hash_table`:return Qb(e,t,n,r);case`custom`:let a=xy(e.op);if(a&&a.customExecutor)return a.customExecutor(new kb(e,t,n));throw TypeError(`Custom op ${e.op} is not registered.`);default:throw TypeError(`Unknown op '${e.op}'. File an issue at https://github.com/tensorflow/tfjs/issues so we can add it, or register a custom execution with tf.registerOp()`)}})(e,t,n);return Te(a)?a.then(e=>[].concat(e)):[].concat(a)}var dx=class{constructor(e={},t={},n={},r={},i){this.weightMap=e,this.tensorArrayMap=t,this.tensorListMap=n,this.functionMap=r,this.parseNodeNameCache=i,this.rootContext={id:0,frameName:``,iterationId:0},this.contexts=[this.rootContext],this.lastId=0,this.generateCurrentContextIds()}newFrame(e,t){return{id:e,frameName:t,iterationId:0}}set currentContext(e){this.contexts!==e&&(this.contexts=e,this.generateCurrentContextIds())}get currentContext(){return this.contexts}get currentContextId(){return this._currentContextIds[0]}get currentContextIds(){return this._currentContextIds}generateCurrentContextIds(){let e=[];for(let t=0;t<this.contexts.length-1;t++){let n=this.contexts.slice(0,this.contexts.length-t);e.push(this.contextIdforContexts(n))}e.push(``),this._currentContextIds=e}contextIdforContexts(e){return e?e.map(e=>e.id===0&&e.iterationId===0?``:`${e.frameName}-${e.iterationId}`).join(`/`):``}enterFrame(e){this.contexts&&(this.lastId++,this.contexts=this.contexts.slice(),this.contexts.push(this.newFrame(this.lastId,e)),this._currentContextIds.unshift(this.contextIdforContexts(this.contexts)))}exitFrame(){if(this.contexts&&this.contexts.length>1)this.contexts=this.contexts.slice(),this.contexts.splice(-1),this.currentContextIds.shift();else throw Error(`Cannot exit frame, the context is empty`)}nextIteration(){if(this.contexts&&this.contexts.length>0){this.contexts=this.contexts.slice(),this.lastId++;let e=Object.assign({},this.contexts[this.contexts.length-1]);e.iterationId+=1,e.id=this.lastId,this.contexts.splice(-1,1,e),this._currentContextIds.splice(0,1,this.contextIdforContexts(this.contexts))}else throw Error(`Cannot increase frame iteration, the context is empty`)}getWeight(e){return this.weightMap[e]}addTensorArray(e){this.tensorArrayMap[e.id]=e}getTensorArray(e){return this.tensorArrayMap[e]}addTensorList(e){this.tensorListMap[e.id]=e}getTensorList(e){return this.tensorListMap[e]}dispose(e){for(let t in this.tensorArrayMap)this.tensorArrayMap[t].clearAndClose(e);for(let t in this.tensorListMap)this.tensorListMap[t].clearAndClose(e)}};function fx(e,t,n,r){let i=new Set,a=[],o=null,s=null,c=new Set,l=new Set(Object.keys(e).map(e=>Ey(e)[0]));r||=[];let u=new Set(r.map(e=>Ey(e.name)[0])),d=[...t];for(;d.length>0;){let e=d.pop();if((xx(e)||Sx(e)||Cx(e))&&(o??(o=e,s=o.children.map(e=>e.name).filter(e=>i.has(e)))),i.add(e.name),n[e.name]==null&&!l.has(e.name)&&!u.has(e.name)){if(e.inputs.length===0){a.push(e.name);continue}e.inputs.forEach(e=>{c.has(e.name)||(c.add(e.name),d.push(e))})}}return{inputs:e,outputs:t,usedNodes:i,missingInputs:a,dynamicNode:o,syncInputs:s}}function px(e,t){let{usedNodes:n,inputs:r}=t,i=Object.keys(r).map(e=>Ey(e)[0]).map(t=>e.nodes[t]),a=e.initNodes||[],o=e=>n.has(typeof e==`string`?e:e.name);function s(e){return[...new Map(e.map(e=>[e.name,e])).values()]}let c=s([...i,...e.weights,...a]).filter(o),l=s([...c,...Object.values(e.nodes)]).filter(o),u=new Map(l.map(e=>[e.name,e])),d={};for(let e of l){d[e.name]=d[e.name]||0;for(let t of e.children)o(t)||(d[t.name]=1/0),d[t.name]=(d[t.name]||0)+1}let f=Object.entries(d).filter(([,e])=>e===0).map(([e])=>e),p=[...f];for(;f.length>0;){let e=f.pop(),t=u.get(e);for(let e of t.children.filter(o))--d[e.name]===0&&(p.push(e.name),f.push(e.name))}let m=mx(p.map(e=>u.get(e)),c);return gx(m,c),m}function mx(e,t){let n=new Map(e.map(e=>[e.name,e])),r=t.map(e=>e.name),i=new Set(r);for(;r.length>0;){let e=r.pop(),t=n.get(e);for(let e of t.children)!n.has(e.name)||i.has(e.name)||(i.add(e.name),r.push(e.name))}return e.filter(e=>i.has(e.name))}var hx=class extends Error{constructor(e){super(`NodesExecutionOrderError: ${e}`)}};function gx(e,t){let n=new Map(e.map((e,t)=>[e.name,t])),r=new Set(t.map(e=>e.name)),i=e=>r.has(typeof e==`string`?e:e.name),a=new Set(e.map(e=>e.name)),o=e=>a.has(typeof e==`string`?e:e.name);for(let t of e){for(let e of t.children.filter(o)){if(!n.has(e.name))throw new hx(`Child ${e.name} of node ${t.name} is unreachable.`);if(n.get(t.name)>n.get(e.name))throw new hx(`Node ${t.name} is scheduled to run after its child ${e.name}.`)}if(!i(t))for(let e of t.inputs){if(!n.has(e.name))throw new hx(`Input ${e.name} of node ${t.name} is unreachable.`);if(n.get(e.name)>n.get(t.name))throw new hx(`Node ${t.name} is scheduled to run before its input ${e.name}.`)}}}function _x(e){let t=new Map(e.map((e,t)=>[e.name,t])),n=2**53-1,r=e.map((e,t)=>xx(e)?n:t),i=e=>r[t.get(e.name)]??-1,a=e.map((e,t)=>e.children.map(i).reduce((e,t)=>Math.max(e,t),r[t])),o=new Map;for(let t=0;t<e.length;++t){let r=a[t];if(r===n)continue;let i=e[t],s=e[r];o.has(s.name)||o.set(s.name,[]),o.get(s.name).push(i)}return o}var vx=new Set([`Switch`,`Merge`,`Enter`,`Exit`,`NextIteration`,`StatelessIf`,`StatelessWhile`,`if`,`While`]),yx=new Set([`NonMaxSuppressionV2`,`NonMaxSuppressionV3`,`NonMaxSuppressionV5`,`Where`]),bx=new Set([`HashTable`,`HashTableV2`,`LookupTableImport`,`LookupTableImportV2`,`LookupTableFind`,`LookupTableFindV2`,`LookupTableSize`,`LookupTableSizeV2`]);function xx(e){return vx.has(e.op)}function Sx(e){return yx.has(e.op)}function Cx(e){return bx.has(e.op)}var wx=class e{get weightIds(){return this.parent?this.parent.weightIds:this._weightIds}get functionExecutorMap(){return this.parent?this.parent.functionExecutorMap:this._functionExecutorMap}get weightMap(){return this.parent?this.parent.weightMap:this._weightMap}set weightMap(e){let t=Object.keys(e).map(t=>e[t].map(e=>e.id));this._weightIds=[].concat(...t),this._weightMap=e}set resourceManager(e){this._resourceManager=e}get inputs(){return this._inputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get outputs(){return this._outputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get inputNodes(){return this._inputs.map(e=>e.signatureKey||e.name)}get outputNodes(){return this._outputs.map(e=>{let t=e.signatureKey||e.name;return e.defaultOutput?`${t}:${e.defaultOutput}`:t})}get functions(){return Object.keys(this._functions).reduce((e,t)=>(e[t]=this._functions[t].signature,e),{})}constructor(t,n){this.graph=t,this.parent=n,this.compiledMap=new Map,this.parseNodeNameCache=new Map,this._weightMap={},this.SEPARATOR=`,`,this._functions={},this._functionExecutorMap={},this.keepIntermediateTensors=!1,this._outputs=t.outputs,this._inputs=t.inputs,this._initNodes=t.initNodes,this._signature=t.signature,this._functions=t.functions,t.functions!=null&&Object.keys(t.functions).forEach(n=>{this._functionExecutorMap[n]=new e(t.functions[n],this)})}getCompilationKey(e,t){let n=e.map(e=>e.name).sort(),r=t.map(e=>e.name).sort();return n.join(this.SEPARATOR)+`--`+r.join(this.SEPARATOR)}compile(e,t){let n=fx(e,t,this.weightMap,this._initNodes),{missingInputs:r,dynamicNode:i,syncInputs:a}=n;if(i!=null)throw Error(`This execution contains the node '${i.name}', which has the dynamic op '${i.op}'. Please use model.executeAsync() instead. Alternatively, to avoid the dynamic ops, specify the inputs [${a}]`);if(r.length>0){let n=t.map(e=>e.name);throw Error(`Cannot compute the outputs [${n}] from the provided inputs [${Object.keys(e)}]. Missing the following inputs: [${r}]`)}let o=px(this.graph,n);return{orderedNodes:o,nodeLiveUntilMap:_x(o)}}cloneAndKeepTensor(e){if(e==null)return null;let t=e.clone();return pa(t),t}cloneTensorList(e){return e?e.map(e=>this.cloneAndKeepTensor(e)):null}cloneTensorMap(e){return Object.fromEntries(Object.entries(e).map(([e,t])=>[e,this.cloneTensorList(t)]))}execute(e,t){this.disposeIntermediateTensors(),e=this.mapInputs(e);let n=Object.keys(e).sort();this.checkInputs(e),this.checkInputShapeAndType(e),t=this.mapOutputs(t),this.checkOutputs(t);let r=n.map(e=>this.graph.nodes[Ey(e)[0]]),i=t.map(e=>Ey(e)[0]),a=new Set(i),o=i.map(e=>this.graph.nodes[e]);o.length===0&&(o=this._outputs);let s=this.getCompilationKey(r,o),c=this.compiledMap.get(s);c??(c=this.compile(e,o),this.compiledMap.set(s,c));try{this.keepIntermediateTensors=O().getBool(`KEEP_INTERMEDIATE_TENSORS`)}catch(e){this.keepIntermediateTensors=!1,console.warn(e.message)}let l={},u={};return P(()=>{let n=new dx(this.weightMap,l,u,this.functionExecutorMap,this.parseNodeNameCache),r=Object.assign({},this.weightMap);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap)),Object.keys(e).forEach(t=>{let[i,a]=Ey(t,n),o=[];o[a]=e[t],r[i]=o,this.keepIntermediateTensors&&(this.clonedTensorsMap[i]=this.cloneTensorList(o))});let i=this.getFrozenTensorIds(r),{orderedNodes:o,nodeLiveUntilMap:s}=c;for(let e of o){if(r[e.name])continue;let t=ux(e,r,n,this._resourceManager);if(Te(t))throw Error(`The execution of the op '${e.op}' returned a promise. Please use model.executeAsync() instead.`);r[e.name]=t,this.keepIntermediateTensors&&(this.clonedTensorsMap[e.name]=this.cloneTensorList(t)),this.checkTensorForDisposalWithNodeLiveUntilInfo(e,r,n,i,a,s.get(e.name))}return this.parent??n.dispose(i),t.map(e=>Sy(e,r,n))})}getFrozenTensorIds(e){let t=[].concat.apply([],Object.keys(e).map(t=>e[t]).map(e=>e.map(e=>e.id)));return new Set(t)}checkTensorForDisposal(e,t,n,r,i,a,o){if(!(xx(t)||a.has(e))){for(let r of n[e])r!=null&&(o[r.id]=(o[r.id]||0)+t.children.length);for(let e of t.inputs){if(xx(e))continue;let t=Cy(e.name,n,r);if(t!=null)for(let e of t){if(!e||e.kept||i.has(e.id))continue;let t=o[e.id];t===1?(e.dispose(),delete o[e.id]):t!=null&&o[e.id]--}}}}checkTensorForDisposalWithNodeLiveUntilInfo(e,t,n,r,i,a){function o(e){return xx(e)||i.has(e.name)}if(!(xx(e)||a==null))for(let e of a){if(o(e))continue;let i=Cy(e.name,t,n);for(let e of i)!e||e.kept||r.has(e.id)||e.dispose()}}async executeAsync(e,t){return this._executeAsync(e,t)}disposeIntermediateTensors(){this.clonedTensorsMap&&=(Object.values(this.clonedTensorsMap).forEach(e=>{for(let t of e)t&&!t.isDisposed&&t.dispose()}),null)}getIntermediateTensors(){return this.clonedTensorsMap}async _executeAsync(e,t,n=!1,r={},i={}){this.disposeIntermediateTensors(),n||(e=this.mapInputs(e),this.checkInputs(e),this.checkInputShapeAndType(e),t=this.mapOutputs(t),this.checkOutputs(t));try{this.keepIntermediateTensors=O().getBool(`KEEP_INTERMEDIATE_TENSORS`)}catch(e){this.keepIntermediateTensors=!1,console.warn(e.message)}let a=new dx(this.weightMap,r,i,this.functionExecutorMap,this.parseNodeNameCache);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap));let o=await this.executeWithControlFlow(e,a,t,n),s=t.map(e=>Sy(e,o,a)),c=s.map(e=>e.id),l=Object.keys(e).map(t=>e[t].id),u=new Set([...c,...l,...this.weightIds]);return Object.values(o).forEach(e=>{e.forEach(e=>{e&&!e.isDisposed&&!u.has(e.id)&&e.dispose()})}),this.parent??a.dispose(u),s}async executeFunctionAsync(e,t,n){let r=e.reduce((e,t,n)=>(e[this.inputs[n].name]=t,e),{});return this._executeAsync(r,this.outputNodes,!0,t,n)}async executeWithControlFlow(e,t,n,r){let i=Object.keys(e),a=i.map(e=>this.graph.nodes[Ey(e)[0]]),o=n.map(e=>Ey(e)[0]),s=new Set(o),c=o.map(e=>this.graph.nodes[e]);c.length===0&&(c=this._outputs);let{usedNodes:l,missingInputs:u,dynamicNode:d,syncInputs:f}=fx(e,c,this.weightMap,this._initNodes),p=[...a,...this.graph.weights,...this._initNodes||[]].map(e=>({node:e,contexts:t.currentContext})),m=Object.assign({},this.weightMap);Object.keys(e).forEach(t=>{let[n,r]=Ey(t),i=[];i[r]=e[t],m[n]=i});let h={},g=this.getFrozenTensorIds(m),_={};for(;p.length>0;){let e=this.processStack(a,p,t,m,_,g,s,h,l);await Promise.all(e)}d==null&&!r&&console.warn(`This model execution did not contain any nodes with control flow or dynamic output shapes. You can use model.execute() instead.`);let v=c.filter(e=>!xx(e)&&!Sy(e.name,m,t)).map(e=>e.name);if(v.length>0){let e=``;throw d!=null&&(e=`Alternatively, to avoid the dynamic ops, use model.execute() and specify the inputs [${f}]`),Error(`Cannot compute the outputs [${v}] from the provided inputs [${i}]. Consider providing the following inputs: [${u}]. ${e}`)}return m}processStack(e,t,n,r,i,a,o,s,c){let l=[];for(;t.length>0;){let e=t.pop();n.currentContext=e.contexts;let u=``;if(e.node.op===`Enter`&&J(`isConstant`,e.node,r,n)&&([u]=wy(e.node.name,n)),r[e.node.name]==null){let d=ux(e.node,r,n,this._resourceManager);u||([u]=wy(e.node.name,n));let f=n.currentContext;Te(d)?l.push(d.then(l=>(r[u]=l,this.keepIntermediateTensors&&(this.clonedTensorsMap[u]=this.cloneTensorList(l)),n.currentContext=f,this.checkTensorForDisposal(u,e.node,r,n,a,o,s),this.processChildNodes(e.node,t,n,r,i,c),l))):(r[u]=d,this.keepIntermediateTensors&&(this.clonedTensorsMap[u]=this.cloneTensorList(d)),this.checkTensorForDisposal(u,e.node,r,n,a,o,s),this.processChildNodes(e.node,t,n,r,i,c))}else this.processChildNodes(e.node,t,n,r,i,c)}return l}processChildNodes(e,t,n,r,i,a){e.children.forEach(e=>{let[o]=wy(e.name,n);i[o]||!a.has(e.name)||(e.op===`Merge`?e.inputNames.some(e=>!!Sy(e,r,n))&&(i[o]=!0,t.push({contexts:n.currentContext,node:e})):e.inputNames.every(e=>!!Sy(e,r,n))&&(i[o]=!0,t.push({contexts:n.currentContext,node:e})))})}dispose(){Object.keys(this.weightMap).forEach(e=>this.weightMap[e].forEach(e=>e.dispose()))}checkInputShapeAndType(e){Object.keys(e).forEach(t=>{let n=e[t],[r]=Ey(t),i=this.graph.nodes[r];if(i.attrParams.shape&&i.attrParams.shape.value){let e=i.attrParams.shape.value;y(e.length===n.shape.length&&n.shape.every((t,n)=>e[n]===-1||e[n]===t),()=>`The shape of dict['${i.name}'] provided in model.execute(dict) must be [${e}], but was [${n.shape}]`)}i.attrParams.dtype&&i.attrParams.dtype.value&&y(n.dtype===i.attrParams.dtype.value,()=>`The dtype of dict['${i.name}'] provided in model.execute(dict) must be ${i.attrParams.dtype.value}, but was ${n.dtype}`)})}mapInputs(e){let t={};for(let n in e){let r=this._signature?.inputs?.[n];r==null?t[n]=e[n]:t[r.name]=e[n]}return t}checkInputs(e){let t=Object.keys(e).filter(e=>{let[t]=Ey(e);return this.graph.nodes[t]==null});if(t.length>0)throw Error(`The dict provided in model.execute(dict) has keys: [${t}] that are not part of graph`)}mapOutputs(e){return e.map(e=>{let t=this._signature?.outputs?.[e];return t==null?e:t.name},{})}checkOutputs(e){e.forEach(e=>{let[t]=Ey(e);if(!this.graph.nodes[t])throw Error(`The output '${e}' is not found in the graph`)})}},Tx=class{constructor(e={},t={}){this.hashTableNameToHandle=e,this.hashTableMap=t}addHashTable(e,t){this.hashTableNameToHandle[e]=t.handle,this.hashTableMap[t.id]=t}getHashTableHandleByName(e){return this.hashTableNameToHandle[e]}getHashTableById(e){return this.hashTableMap[e]}dispose(){for(let e in this.hashTableMap)this.hashTableMap[e].clearAndClose(),delete this.hashTableMap[e];for(let e in this.hashTableNameToHandle)this.hashTableNameToHandle[e].dispose(),delete this.hashTableNameToHandle[e]}},Ex=`?tfjs-format=file`,Dx=`model.json`,Ox=class{get modelVersion(){return this.version}get inputNodes(){return this.executor.inputNodes}get outputNodes(){return this.executor.outputNodes}get inputs(){return this.executor.inputs}get outputs(){return this.executor.outputs}get weights(){return this.executor.weightMap}get metadata(){return this.artifacts.userDefinedMetadata}get modelSignature(){return this.signature}get modelStructuredOutputKeys(){return this.structuredOutputKeys}constructor(e,t={},n=F_){this.modelUrl=e,this.loadOptions=t,this.version=`n/a`,this.io=n,t??(this.loadOptions={}),this.resourceManager=new Tx}findIOHandler(){let e=this.modelUrl;if(e.load!=null)this.handler=e;else if(this.loadOptions.requestInit!=null)this.handler=this.io.browserHTTPRequest(e,this.loadOptions);else{let t=this.io.getLoadHandlers(e,this.loadOptions);if(t.length===0)t.push(this.io.browserHTTPRequest(e,this.loadOptions));else if(t.length>1)throw Error(`Found more than one (${t.length}) load handlers for URL '${[e]}'`);this.handler=t[0]}}load(){if(this.findIOHandler(),this.handler.load==null)throw Error("Cannot proceed with model loading because the IOHandler provided does not have the `load` method implemented.");let e=this.handler.load();return Te(e)?e.then(e=>e.getWeightStream==null?this.loadSync(e):this.loadStreaming(e)):this.loadSync(e)}loadSync(e){let t=this.io.decodeWeights(e.weightData,e.weightSpecs);return this.loadWithWeightMap(e,t)}async loadStreaming(e){if(e.getWeightStream==null)throw Error(`Model artifacts missing streamWeights function`);let t=await wa(e.getWeightStream(),e.weightSpecs);return this.loadWithWeightMap(e,t)}loadWithWeightMap(e,t){this.artifacts=e;let n=this.artifacts.modelTopology,r=this.artifacts.signature;if(this.artifacts.userDefinedMetadata!=null){let e=this.artifacts.userDefinedMetadata;e.signature!=null&&(r=e.signature),e.structuredOutputKeys!=null&&(this.structuredOutputKeys=e.structuredOutputKeys)}if(this.signature=r,this.version=`${n.versions.producer}.${n.versions.minConsumer}`,this.executor=new wx(pb.Instance.transformGraph(n,this.signature)),this.executor.weightMap=this.convertTensorMapToTensorsMap(t),this.executor.resourceManager=this.resourceManager,e.modelInitializer!=null&&e.modelInitializer.node!=null){let t=pb.Instance.transformGraph(e.modelInitializer);this.initializer=new wx(t),this.initializer.weightMap=this.executor.weightMap,this.initializer.resourceManager=this.resourceManager,this.initializerSignature=e.initializerSignature}return!0}async save(e,t){if(typeof e==`string`){let t=this.io.getSaveHandlers(e);if(t.length===0)throw Error(`Cannot find any save handlers for URL '${e}'`);if(t.length>1)throw Error(`Found more than one (${t.length}) save handlers for URL '${e}'`);e=t[0]}if(e.save==null)throw Error("GraphModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");return e.save(this.artifacts)}addStructuredOutputNames(e){if(this.structuredOutputKeys){let t=e instanceof Ei?[e]:e,n={};return t.forEach((e,t)=>n[this.structuredOutputKeys[t]]=e),n}return e}predict(e,t){let n=this.execute(e,this.outputNodes);return this.addStructuredOutputNames(n)}async predictAsync(e,t){let n=await this.executeAsync(e,this.outputNodes);return this.addStructuredOutputNames(n)}normalizeInputs(e){if(!(e instanceof Ei)&&!Array.isArray(e)){let t=this.signature?.inputs;if(t!=null)for(let n in t){let r=t[n];r.resourceId!=null&&(e[n]=this.resourceIdToCapturedInput[r.resourceId])}return e}e=Array.isArray(e)?e:[e];let t=Object.keys(this.resourceIdToCapturedInput).length;if(e.length+t!==this.inputNodes.length)throw Error(`Input tensor count mismatch, the graph model has ${this.inputNodes.length-t} non-resource placeholders, while there are ${e.length} input tensors provided.`);let n=0;return this.inputNodes.reduce((t,r)=>{let i=this.signature?.inputs?.[r]?.resourceId;return i==null?t[r]=e[n++]:t[r]=this.resourceIdToCapturedInput[i],t},{})}normalizeOutputs(e){return e||=this.outputNodes,Array.isArray(e)?e:[e]}executeInitializerGraph(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.execute({},[]):this.initializer.execute({},Object.keys(this.initializerSignature.outputs))}async executeInitializerGraphAsync(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.executeAsync({},[]):this.initializer.executeAsync({},Object.keys(this.initializerSignature.outputs))}setResourceIdToCapturedInput(e){if(this.resourceIdToCapturedInput={},this.initializerSignature){let t=this.initializerSignature.outputs,n=Object.keys(t);for(let r=0;r<n.length;r++){let i=t[n[r]];this.resourceIdToCapturedInput[i.resourceId]=e[r]}}}execute(e,t){this.resourceIdToCapturedInput??this.setResourceIdToCapturedInput(this.executeInitializerGraph()),e=this.normalizeInputs(e),t=this.normalizeOutputs(t);let n=this.executor.execute(e,t);return n.length>1?n:n[0]}async executeAsync(e,t){this.resourceIdToCapturedInput??this.setResourceIdToCapturedInput(await this.executeInitializerGraphAsync()),e=this.normalizeInputs(e),t=this.normalizeOutputs(t);let n=await this.executor.executeAsync(e,t);return n.length>1?n:n[0]}getIntermediateTensors(){return this.executor.getIntermediateTensors()}disposeIntermediateTensors(){this.executor.disposeIntermediateTensors()}convertTensorMapToTensorsMap(e){return Object.keys(e).reduce((t,n)=>(t[n]=[e[n]],t),{})}dispose(){this.executor.dispose(),this.initializer&&(this.initializer.dispose(),this.resourceIdToCapturedInput&&F(this.resourceIdToCapturedInput)),this.resourceManager.dispose()}};async function kx(e,t={},n=F_){if(e==null)throw Error(`modelUrl in loadGraphModel() cannot be null. Please provide a url or an IOHandler that loads the model`);t??={},t.fromTFHub&&typeof e==`string`&&(e=Ax(e));let r=new Ox(e,t,n);return await r.load(),r}function Ax(e){return e.endsWith(`/`)||(e+=`/`),`${e}${Dx}${Ex}`}var jx=function(){return jx=Object.assign||function(e){for(var t,n=1,r=arguments.length;n<r;n++)for(var i in t=arguments[n])Object.prototype.hasOwnProperty.call(t,i)&&(e[i]=t[i]);return e},jx.apply(this,arguments)};function Mx(e,t,n,r){return new(n||=Promise)((function(i,a){function o(e){try{c(r.next(e))}catch(e){a(e)}}function s(e){try{c(r.throw(e))}catch(e){a(e)}}function c(e){var t;e.done?i(e.value):(t=e.value,t instanceof n?t:new n((function(e){e(t)}))).then(o,s)}c((r=r.apply(e,t||[])).next())}))}function Nx(e,t){var n,r,i,a,o={label:0,sent:function(){if(1&i[0])throw i[1];return i[1]},trys:[],ops:[]};return a={next:s(0),throw:s(1),return:s(2)},typeof Symbol==`function`&&(a[Symbol.iterator]=function(){return this}),a;function s(s){return function(c){return function(s){if(n)throw TypeError(`Generator is already executing.`);for(;a&&(a=0,s[0]&&(o=0)),o;)try{if(n=1,r&&(i=2&s[0]?r.return:s[0]?r.throw||((i=r.return)&&i.call(r),0):r.next)&&!(i=i.call(r,s[1])).done)return i;switch(r=0,i&&(s=[2&s[0],i.value]),s[0]){case 0:case 1:i=s;break;case 4:return o.label++,{value:s[1],done:!1};case 5:o.label++,r=s[1],s=[0];continue;case 7:s=o.ops.pop(),o.trys.pop();continue;default:if(i=o.trys,!((i=i.length>0&&i[i.length-1])||s[0]!==6&&s[0]!==2)){o=0;continue}if(s[0]===3&&(!i||s[1]>i[0]&&s[1]<i[3])){o.label=s[1];break}if(s[0]===6&&o.label<i[1]){o.label=i[1],i=s;break}if(i&&o.label<i[2]){o.label=i[2],o.ops.push(s);break}i[2]&&o.ops.pop(),o.trys.pop();continue}s=t.call(e,o)}catch(e){s=[6,e],r=0}finally{n=i=0}if(5&s[0])throw s[1];return{value:s[0]?s[1]:void 0,done:!0}}([s,c])}}}var Px=[`wrist`,`thumb_cmc`,`thumb_mcp`,`thumb_ip`,`thumb_tip`,`index_finger_mcp`,`index_finger_pip`,`index_finger_dip`,`index_finger_tip`,`middle_finger_mcp`,`middle_finger_pip`,`middle_finger_dip`,`middle_finger_tip`,`ring_finger_mcp`,`ring_finger_pip`,`ring_finger_dip`,`ring_finger_tip`,`pinky_finger_mcp`,`pinky_finger_pip`,`pinky_finger_dip`,`pinky_finger_tip`],Fx={runtime:`mediapipe`,maxHands:2,modelType:`full`},Ix=function(){function e(e){var t,n=this;this.width=0,this.height=0,this.selfieMode=!1,this.handsSolution=new(void 0)({locateFile:function(t,n){return e.solutionPath?e.solutionPath.replace(/\/+$/,``)+`/`+t:n+`/`+t}}),t=e.modelType===`lite`?0:1,this.handsSolution.setOptions({modelComplexity:t,selfieMode:this.selfieMode,maxNumHands:e.maxHands}),this.handsSolution.onResults((function(e){if(n.height=e.image.height,n.width=e.image.width,n.hands=[],e.multiHandLandmarks!==null)for(var t=e.multiHandedness,r=e.multiHandLandmarks,i=e.multiHandWorldLandmarks,a=0;a<t.length;a++)n.hands.push(jx(jx({},n.translateOutput(r[a],i[a])),{score:t[a].score,handedness:t[a].label}))}))}return e.prototype.translateOutput=function(e,t){var n=this;return{keypoints:e.map((function(e,t){return{x:e.x*n.width,y:e.y*n.height,score:e.visibility,name:Px[t]}})),keypoints3D:t.map((function(e,t){return{x:e.x,y:e.y,z:e.z,score:e.visibility,name:Px[t]}}))}},e.prototype.estimateHands=function(e,t){return Mx(this,void 0,void 0,(function(){var n,r;return Nx(this,(function(i){switch(i.label){case 0:return t&&t.flipHorizontal&&t.flipHorizontal!==this.selfieMode&&(this.selfieMode=t.flipHorizontal,this.handsSolution.setOptions({selfieMode:this.selfieMode})),e instanceof Ei?(r=ImageData.bind,[4,B_(e)]):[3,2];case 1:return n=new(r.apply(ImageData,[void 0,i.sent(),e.shape[1],e.shape[0]])),[3,3];case 2:n=e,i.label=3;case 3:return e=n,[4,this.handsSolution.send({image:e})];case 4:return i.sent(),[2,this.hands]}}))}))},e.prototype.dispose=function(){this.handsSolution.close()},e.prototype.reset=function(){this.handsSolution.reset(),this.width=0,this.height=0,this.hands=null,this.selfieMode=!1},e.prototype.initialize=function(){return this.handsSolution.initialize()},e}();function Lx(e){return Mx(this,void 0,void 0,(function(){var t,n;return Nx(this,(function(r){switch(r.label){case 0:return t=function(e){if(e==null)return jx({},Fx);var t=jx({},e);return t.runtime=`mediapipe`,t.maxHands??=Fx.maxHands,t.modelType??=Fx.modelType,t}(e),[4,(n=new Ix(t)).initialize()];case 1:return r.sent(),[2,n]}}))}))}function Rx(e){return e.width*e.height}function zx(e){var t=e.xCenter-e.width/2,n=t+e.width,r=e.yCenter-e.height/2;return{xMin:t,xMax:n,yMin:r,yMax:r+e.height,width:e.width,height:e.height}}function Bx(e,t){var n=zx(e),r=zx(t);if(!function(e,t){return!(e.xMax<t.xMin||t.xMax<e.xMin||e.yMax<t.yMin||t.yMax<e.yMin)}(n,r))return 0;var i=Rx(function(e,t){var n=Math.max(e.xMin,t.xMin),r=Math.min(e.xMax,t.xMax),i=Math.max(e.yMin,t.yMin),a=Math.min(e.yMax,t.yMax);return{xMin:n,xMax:r,yMin:i,yMax:a,width:Math.max(r-n,0),height:Math.max(a-i,0)}}(n,r)),a=Rx(n)+Rx(r)-i;return a>0?i/a:0}function Vx(e,t,n,r){var i=e.width,a=e.height,o=r?-1:1,s=Math.cos(e.rotation),c=Math.sin(e.rotation),l=e.xCenter,u=e.yCenter,d=1/t,f=1/n,p=Array(16);return p[0]=i*s*o*d,p[1]=-a*c*d,p[2]=0,p[3]=(-.5*i*s*o+.5*a*c+l)*d,p[4]=i*c*o*f,p[5]=a*s*f,p[6]=0,p[7]=(-.5*a*s-.5*i*c*o+u)*f,p[8]=0,p[9]=0,p[10]=i*d,p[11]=0,p[12]=0,p[13]=0,p[14]=0,p[15]=1,function(e){if(e.length!==16)throw Error(`Array length must be 16 but got `+e.length);return[[e[0],e[1],e[2],e[3]],[e[4],e[5],e[6],e[7]],[e[8],e[9],e[10],e[11]],[e[12],e[13],e[14],e[15]]]}(p)}function Hx(e){return e instanceof Ei?{height:e.shape[0],width:e.shape[1]}:{height:e.height,width:e.width}}function Ux(e){return e-2*Math.PI*Math.floor((e+Math.PI)/(2*Math.PI))}function Wx(e){return e instanceof Ei?e:V_(e)}function Gx(e,t){y(e.width!==0,(function(){return t+` width cannot be 0.`})),y(e.height!==0,(function(){return t+` height cannot be 0.`}))}function Kx(e,t){var n=function(e,t,n,r){var i=t-e,a=r-n;if(i===0)throw Error(`Original min and max are both `+e+`, range cannot be 0.`);var o=a/i;return{scale:o,offset:n-e*o}}(0,255,t[0],t[1]);return P((function(){return R(B(e,n.scale),n.offset)}))}function qx(e,t,n){var r=t.outputTensorSize,i=t.keepAspectRatio,a=t.borderMode,o=t.outputTensorFloatRange,s=Hx(e),c=function(e,t){return t?{xCenter:t.xCenter*e.width,yCenter:t.yCenter*e.height,width:t.width*e.width,height:t.height*e.height,rotation:t.rotation}:{xCenter:.5*e.width,yCenter:.5*e.height,width:e.width,height:e.height,rotation:0}}(s,n),l=function(e,t,n){if(n===void 0&&(n=!1),!n)return{top:0,left:0,right:0,bottom:0};var r=t.height,i=t.width;Gx(t,`targetSize`),Gx(e,`roi`);var a,o,s=r/i,c=e.height/e.width,l=0,u=0;return s>c?(a=e.width,o=e.width*s,u=(1-c/s)/2):(a=e.height/s,o=e.height,l=(1-s/c)/2),e.width=a,e.height=o,{top:u,left:l,right:l,bottom:u}}(c,r,i),u=Vx(c,s.width,s.height,!1);return{imageTensor:P((function(){var t=Wx(e),n=Kp(function(e,t,n){return Gx(n,`inputResolution`),[1/n.width*e[0][0]*t.width,1/n.height*e[0][1]*t.width,e[0][3]*t.width,1/n.width*e[1][0]*t.height,1/n.height*e[1][1]*t.height,e[1][3]*t.height,0,0]}(u,s,r),[1,8]),i=a===`zero`?`constant`:`nearest`,c=Ug.transform(Xl(L(t,`float32`)),n,`bilinear`,i,0,[r.height,r.width]);return o==null?c:Kx(c,o)})),padding:l,transformationMatrix:u}}function Jx(e,t,n,r){return r===1?.5*(e+t):e+(t-e)*n/(r-1)}function Yx(e){return{xCenter:e.xMin+e.width/2,yCenter:e.yMin+e.height/2,width:e.width,height:e.height}}function Xx(e){var t=e.relativeKeypoints;if(t.length<=1)throw Error(`2 or more keypoints required to calculate a rect.`);var n=Number.MAX_VALUE,r=Number.MAX_VALUE,i=Number.MIN_VALUE,a=Number.MIN_VALUE;return t.forEach((function(e){n=Math.min(n,e.x),i=Math.max(i,e.x),r=Math.min(r,e.y),a=Math.max(a,e.y)})),{xCenter:(n+i)/2,yCenter:(r+a)/2,width:i-n,height:a-r}}function Zx(e,t,n,r,i){var a=n===`rect`?function(e,t,n){var r,i=e.locationData;if(t===`boundingbox`)r=Yx(i.boundingBox);else{r=Xx(i);var a=n.width,o=n.height;r.xCenter=Math.round(r.xCenter*a),r.yCenter=Math.round(r.yCenter*o),r.width=Math.round(r.width*a),r.height=Math.round(r.height*o)}return r}(e,t,r):function(e,t){var n=e.locationData;return t===`boundingbox`?Yx(n.relativeBoundingBox):Xx(n)}(e,t);return i&&(a.rotation=function(e,t,n){var r,i=e.locationData,a=n.rotationVectorStartKeypointIndex,o=n.rotationVectorEndKeypointIndex;r=n.rotationVectorTargetAngle?n.rotationVectorTargetAngle:Math.PI*n.rotationVectorTargetAngleDegree/180;var s=i.relativeKeypoints[a].x*t.width,c=i.relativeKeypoints[a].y*t.height,l=i.relativeKeypoints[o].x*t.width,u=i.relativeKeypoints[o].y*t.height;return Ux(r-Math.atan2(-(u-c),l-s))}(e,r,i)),a}function Qx(e){return P((function(){var t=function(e){return P((function(){return[U(e,[0,0,0],[1,-1,1]),U(e,[0,0,1],[1,-1,-1])]}))}(e),n=t[0],r=t[1];return{boxes:Ip(r),logits:Ip(n)}}))}function $x(e,t,n,r){return Mx(this,void 0,void 0,(function(){var r,i,a,o,s;return Nx(this,(function(c){switch(c.label){case 0:return e.sort((function(e,t){return Math.max.apply(Math,t.score)-Math.max.apply(Math,e.score)})),r=Kp(e.map((function(e){return[e.locationData.relativeBoundingBox.yMin,e.locationData.relativeBoundingBox.xMin,e.locationData.relativeBoundingBox.yMax,e.locationData.relativeBoundingBox.xMax]}))),i=Gp(e.map((function(e){return e.score[0]}))),[4,Ug.nonMaxSuppressionAsync(r,i,t,n)];case 1:return[4,(a=c.sent()).array()];case 2:return o=c.sent(),s=e.filter((function(e,t){return o.indexOf(t)>-1})),F([r,i,a]),[2,s]}}))}))}function eS(e,t){return e.map((function(e){var n=jx(jx({},e),{x:e.x*t.width,y:e.y*t.height});return e.z!=null&&(n.z=e.z*t.width),n}))}function tS(e,t,n){return Mx(this,void 0,void 0,(function(){var r,i,a,o,s;return Nx(this,(function(c){switch(c.label){case 0:return r=e[0],i=e[1],a=function(e,t,n){return P((function(){var r,i,a,o;n.reverseOutputOrder?(i=Ip(U(e,[0,n.boxCoordOffset+0],[-1,1])),r=Ip(U(e,[0,n.boxCoordOffset+1],[-1,1])),o=Ip(U(e,[0,n.boxCoordOffset+2],[-1,1])),a=Ip(U(e,[0,n.boxCoordOffset+3],[-1,1]))):(r=Ip(U(e,[0,n.boxCoordOffset+0],[-1,1])),i=Ip(U(e,[0,n.boxCoordOffset+1],[-1,1])),a=Ip(U(e,[0,n.boxCoordOffset+2],[-1,1])),o=Ip(U(e,[0,n.boxCoordOffset+3],[-1,1]))),i=R(B(z(i,n.xScale),t.w),t.x),r=R(B(z(r,n.yScale),t.h),t.y),n.applyExponentialOnBoxSize?(a=B(Jl(z(a,n.hScale)),t.h),o=B(Jl(z(o,n.wScale)),t.w)):(a=B(z(a,n.hScale),t.h),o=B(z(o,n.wScale),t.h));var s=q(r,z(a,2)),c=q(i,z(o,2)),l=R(r,z(a,2)),u=R(i,z(o,2)),d=Rs([V(s,[n.numBoxes,1]),V(c,[n.numBoxes,1]),V(l,[n.numBoxes,1]),V(u,[n.numBoxes,1])],1);if(n.numKeypoints)for(var f=0;f<n.numKeypoints;++f){var p=n.keypointCoordOffset+f*n.numValuesPerKeypoint,m=void 0,h=void 0;n.reverseOutputOrder?(m=Ip(U(e,[0,p],[-1,1])),h=Ip(U(e,[0,p+1],[-1,1]))):(h=Ip(U(e,[0,p],[-1,1])),m=Ip(U(e,[0,p+1],[-1,1])));var g=R(B(z(m,n.xScale),t.w),t.x),_=R(B(z(h,n.yScale),t.h),t.y);d=Rs([d,V(g,[n.numBoxes,1]),V(_,[n.numBoxes,1])],1)}return d}))}(i,t,n),o=P((function(){var e=r;return n.sigmoidScore?(n.scoreClippingThresh!=null&&(e=gc(r,-n.scoreClippingThresh,n.scoreClippingThresh)),e=Vs(e)):e})),[4,nS(a,o,n)];case 1:return s=c.sent(),F([a,o]),[2,s]}}))}))}function nS(e,t,n){return Mx(this,void 0,void 0,(function(){var r,i,a,o,s,c,l,u,d,f,p,m;return Nx(this,(function(h){switch(h.label){case 0:return r=[],[4,e.data()];case 1:return i=h.sent(),[4,t.data()];case 2:for(a=h.sent(),o=0;o<n.numBoxes;++o)if(!(n.minScoreThresh!=null&&a[o]<n.minScoreThresh||(s=o*n.numCoords,c=rS(i[s+0],i[s+1],i[s+2],i[s+3],a[o],n.flipVertically,o),(l=c.locationData.relativeBoundingBox).width<0||l.height<0))){if(n.numKeypoints>0)for((u=c.locationData).relativeKeypoints=[],d=n.numKeypoints*n.numValuesPerKeypoint,f=0;f<d;f+=n.numValuesPerKeypoint)p=s+n.keypointCoordOffset+f,m={x:i[p+0],y:n.flipVertically?1-i[p+1]:i[p+1]},u.relativeKeypoints.push(m);r.push(c)}return[2,r]}}))}))}function rS(e,t,n,r,i,a,o){return{score:[i],ind:o,locationData:{relativeBoundingBox:{xMin:t,yMin:a?1-n:e,xMax:r,yMax:a?1-e:n,width:r-t,height:n-e}}}}function iS(e,t){return e===`none`?t:function(e){return 1/(1+Math.exp(-e))}(t)}function aS(e,t,n,r){return Mx(this,void 0,void 0,(function(){var i,a,o,s,c,l,u,d;return Nx(this,(function(f){switch(f.label){case 0:return n=n||t.flipHorizontally||!1,r=r||t.flipVertically||!1,i=e.size,a=i/t.numLandmarks,[4,e.data()];case 1:for(o=f.sent(),s=[],c=0;c<t.numLandmarks;++c)l=c*a,(d={x:0,y:0}).x=n?t.inputImageWidth-o[l]:o[l],a>1&&(d.y=r?t.inputImageHeight-o[l+1]:o[l+1]),a>2&&(d.z=o[l+2]),a>3&&(d.score=iS(t.visibilityActivation,o[l+3])),s.push(d);for(u=0;u<s.length;++u)(d=s[u]).x=d.x/t.inputImageWidth,d.y/=t.inputImageHeight,d.z=d.z/t.inputImageWidth/(t.normalizeZ||1);return[2,s]}}))}))}function oS(e,t,n){var r=e.width,i=e.height,a=e.rotation;if(n.rotation==null&&n.rotationDegree==null||(a=function(e,t){return t.rotation==null?t.rotationDegree!=null&&(e+=Math.PI*t.rotationDegree/180):e+=t.rotation,Ux(e)}(a,n)),a===0)e.xCenter+=r*n.shiftX,e.yCenter+=i*n.shiftY;else{var o=(t.width*r*n.shiftX*Math.cos(a)-t.height*i*n.shiftY*Math.sin(a))/t.width,s=(t.width*r*n.shiftX*Math.sin(a)+t.height*i*n.shiftY*Math.cos(a))/t.height;e.xCenter+=o,e.yCenter+=s}if(n.squareLong){var c=Math.max(r*t.width,i*t.height);r=c/t.width,i=c/t.height}else if(n.squareShort){var l=Math.min(r*t.width,i*t.height);r=l/t.width,i=l/t.height}return e.width=r*n.scaleX,e.height=i*n.scaleY,e}function sS(e,t){for(var n=function(e,t){var n=e[0].x*t.width,r=e[0].y*t.height,i=(e[4].x+e[8].x)/2,a=(e[4].y+e[8].y)/2;return i=(i+e[6].x)/2*t.width,a=(a+e[6].y)/2*t.height,Ux(Math.PI/2-Math.atan2(-(a-r),i-n))}(e,t),r=Ux(-n),i=1/0,a=-1/0,o=1/0,s=-1/0,c=0,l=e;c<l.length;c++){var u=(g=l[c]).x,d=g.y;i=Math.min(i,u),a=Math.max(a,u),o=Math.min(o,d),s=Math.max(s,d)}var f=(a+i)/2,p=(s+o)/2;i=1/0,a=-1/0,o=1/0,s=-1/0;for(var m=0,h=e;m<h.length;m++){var g,_=((g=h[m]).x-f)*t.width,v=(g.y-p)*t.height,y=_*Math.cos(r)-v*Math.sin(r),b=_*Math.sin(r)+v*Math.cos(r);i=Math.min(i,y),a=Math.max(a,y),o=Math.min(o,b),s=Math.max(s,b)}var x=(a+i)/2,S=(s+o)/2,C=x*Math.cos(n)-S*Math.sin(n)+t.width*f,w=x*Math.sin(n)+S*Math.cos(n)+t.height*p,ee=(a-i)/t.width,T=(s-o)/t.height;return{xCenter:C/t.width,yCenter:w/t.height,width:ee,height:T,rotation:n}}var cS={reduceBoxesInLowestLayer:!1,interpolatedScaleAspectRatio:1,featureMapHeight:[],featureMapWidth:[],numLayers:4,minScale:.1484375,maxScale:.75,inputSizeHeight:192,inputSizeWidth:192,anchorOffsetX:.5,anchorOffsetY:.5,strides:[8,16,16,16],aspectRatios:[1],fixedAnchorSize:!0},lS={runtime:`tfjs`,modelType:`full`,maxHands:2,detectorModelUrl:`https://tfhub.dev/mediapipe/tfjs-model/handpose_3d/detector/full/1`,landmarkModelUrl:`https://tfhub.dev/mediapipe/tfjs-model/handpose_3d/landmark/full/1`},uS={flipHorizontal:!1,staticImageMode:!1},dS={applyExponentialOnBoxSize:!1,flipVertically:!1,ignoreClasses:[],numClasses:1,numBoxes:2016,numCoords:18,boxCoordOffset:0,keypointCoordOffset:4,numKeypoints:7,numValuesPerKeypoint:2,sigmoidScore:!0,scoreClippingThresh:100,reverseOutputOrder:!0,xScale:192,yScale:192,hScale:192,wScale:192,minScoreThresh:.5},fS=.3,pS={shiftX:0,shiftY:-.5,scaleX:2.6,scaleY:2.6,squareLong:!0},mS={shiftX:0,shiftY:-.1,scaleX:2,scaleY:2,squareLong:!0},hS={outputTensorSize:{width:192,height:192},keepAspectRatio:!0,outputTensorFloatRange:[0,1],borderMode:`zero`},gS={outputTensorSize:{width:224,height:224},keepAspectRatio:!0,outputTensorFloatRange:[0,1],borderMode:`zero`},_S={numLandmarks:21,inputImageWidth:224,inputImageHeight:224,normalizeZ:.4,visibilityActivation:`none`,flipHorizontally:!1,flipVertically:!1},vS={numLandmarks:21,inputImageWidth:1,inputImageHeight:1,visibilityActivation:`none`,flipHorizontally:!1,flipVertically:!1},yS,bS=function(){function e(e,t,n){this.detectorModel=e,this.landmarkModel=t,this.maxHands=n,this.prevHandRectsFromLandmarks=null,this.anchors=function(e){e.reduceBoxesInLowestLayer??=!1,e.interpolatedScaleAspectRatio??=1,e.fixedAnchorSize??=!1;for(var t=[],n=0;n<e.numLayers;){for(var r=[],i=[],a=[],o=[],s=n;s<e.strides.length&&e.strides[s]===e.strides[n];){var c=Jx(e.minScale,e.maxScale,s,e.strides.length);if(s===0&&e.reduceBoxesInLowestLayer)a.push(1),a.push(2),a.push(.5),o.push(.1),o.push(c),o.push(c);else{for(var l=0;l<e.aspectRatios.length;++l)a.push(e.aspectRatios[l]),o.push(c);if(e.interpolatedScaleAspectRatio>0){var u=s===e.strides.length-1?1:Jx(e.minScale,e.maxScale,s+1,e.strides.length);o.push(Math.sqrt(c*u)),a.push(e.interpolatedScaleAspectRatio)}}s++}for(var d=0;d<a.length;++d){var f=Math.sqrt(a[d]);r.push(o[d]/f),i.push(o[d]*f)}var p=0,m=0;if(e.featureMapHeight.length>0)p=e.featureMapHeight[n],m=e.featureMapWidth[n];else{var h=e.strides[n];p=Math.ceil(e.inputSizeHeight/h),m=Math.ceil(e.inputSizeWidth/h)}for(var g=0;g<p;++g)for(var _=0;_<m;++_)for(var v=0;v<r.length;++v){var y={xCenter:(_+e.anchorOffsetX)/m,yCenter:(g+e.anchorOffsetY)/p,width:0,height:0};e.fixedAnchorSize?(y.width=1,y.height=1):(y.width=i[v],y.height=r[v]),t.push(y)}n=s}return t}(cS);var r=Gp(this.anchors.map((function(e){return e.width}))),i=Gp(this.anchors.map((function(e){return e.height}))),a=Gp(this.anchors.map((function(e){return e.xCenter}))),o=Gp(this.anchors.map((function(e){return e.yCenter})));this.anchorTensor={x:a,y:o,w:r,h:i}}return e.prototype.estimateHands=function(e,t){return Mx(this,void 0,void 0,(function(){var n,r,i,a,o,s,c,l,u,d,f,p,m,h,g,_,v,y,b=this;return Nx(this,(function(x){switch(x.label){case 0:return n=function(e){if(e==null)return jx({},uS);var t=jx({},e);return t.flipHorizontal??=uS.flipHorizontal,t.staticImageMode??=uS.staticImageMode,t}(t),e==null?(this.reset(),[2,[]]):(r=Hx(e),i=P((function(){var t=L(Wx(e),`float32`);return n.flipHorizontal&&(t=Ip(Ug.flipLeftRight(Xl(t,0)),[0])),t})),a=this.prevHandRectsFromLandmarks,n.staticImageMode||a==null||a.length<this.maxHands?[4,this.detectPalm(i)]:[3,2]);case 1:return(s=x.sent()).length===0?(this.reset(),i.dispose(),[2,[]]):(c=s.map((function(e){return b.palmDetectionToRoi(e,r)})),o=c,[3,3]);case 2:o=a,x.label=3;case 3:return S=.5,C=[],[o].forEach((function(e){return e.forEach((function(e){(C=C.filter((function(t){return Bx(e,t)<=S}))).push(e)}))})),o=C,[4,Promise.all(o.map((function(e){return b.handLandmarks(e,i)})))];case 4:for(l=x.sent(),u=[],this.prevHandRectsFromLandmarks=[],d=0,f=l;d<f.length;d++)(p=f[d])!=null&&(m=p.landmarks,h=p.worldLandmarks,g=p.handScore,_=p.handedness,this.prevHandRectsFromLandmarks.push(this.handLandmarksToRoi(m,r)),(v=eS(m,r))!=null&&v.forEach((function(e,t){delete e.z,e.name=Px[t]})),(y=h)!=null&&y.forEach((function(e,t){e.name=Px[t]})),u.push({keypoints:v,keypoints3D:y,handedness:_,score:g}));return i.dispose(),[2,u]}var S,C}))}))},e.prototype.dispose=function(){this.detectorModel.dispose(),this.landmarkModel.dispose(),F([this.anchorTensor.x,this.anchorTensor.y,this.anchorTensor.w,this.anchorTensor.h])},e.prototype.reset=function(){this.prevHandRectsFromLandmarks=null},e.prototype.detectPalm=function(e){return Mx(this,void 0,void 0,(function(){var t,n,r,i,a,o,s,c,l,u;return Nx(this,(function(d){switch(d.label){case 0:return t=qx(e,hS),n=t.imageTensor,r=t.padding,i=this.detectorModel.predict(n),a=Qx(i),o=a.boxes,[4,tS([s=a.logits,o],this.anchorTensor,dS)];case 1:return(c=d.sent()).length===0?(F([n,i,s,o]),[2,c]):[4,$x(c,this.maxHands,fS)];case 2:return l=d.sent(),u=function(e,t){e===void 0&&(e=[]);for(var n=t.left,r=t.top,i=t.left+t.right,a=t.top+t.bottom,o=0;o<e.length;o++){var s=e[o],c=s.locationData.relativeBoundingBox,l=(c.xMin-n)/(1-i),u=(c.yMin-r)/(1-a),d=c.width/(1-i),f=c.height/(1-a);c.xMin=l,c.yMin=u,c.width=d,c.height=f,c.xMax=l+d,c.yMax=u+f;var p=s.locationData.relativeKeypoints;p&&p.forEach((function(e){var t=(e.x-n)/(1-i),o=(e.y-r)/(1-a);e.x=t,e.y=o}))}return e}(l,r),F([n,i,s,o]),[2,u]}}))}))},e.prototype.palmDetectionToRoi=function(e,t){return oS(Zx(e,`boundingbox`,`normRect`,t,{rotationVectorStartKeypointIndex:0,rotationVectorEndKeypointIndex:2,rotationVectorTargetAngleDegree:90}),t,pS)},e.prototype.handLandmarks=function(e,t){return Mx(this,void 0,void 0,(function(){var n,r,i,a,o,s,c,l,u,d,f,p,m,h,g,_;return Nx(this,(function(v){switch(v.label){case 0:return n=qx(t,gS,e),r=n.imageTensor,i=n.padding,a=this.landmarkModel.execute(r,[`Identity_2:0`,`Identity_1:0`,`Identity:0`,`Identity_3:0`]),o=a[0],s=a[1],c=a[2],l=a[3],[4,s.data()];case 1:return(u=v.sent()[0])<.5?(F(a),F(r),[2,null]):[4,c.data()];case 2:return d=v.sent()[0],f=d>=.5?`Left`:`Right`,[4,aS(o,_S)];case 3:return p=v.sent(),[4,aS(l,vS)];case 4:return m=v.sent(),h=function(e,t){var n=t.left,r=t.top,i=t.left+t.right,a=t.top+t.bottom;return e.map((function(e){return jx(jx({},e),{x:(e.x-n)/(1-i),y:(e.y-r)/(1-a),z:e.z/(1-i)})}))}(p,i),g=function(e,t,n){n===void 0&&(n={ignoreRotation:!1});for(var r=[],i=0,a=e;i<a.length;i++){var o=a[i],s=o.x-.5,c=o.y-.5,l=n.ignoreRotation?0:t.rotation,u=Math.cos(l)*s-Math.sin(l)*c,d=Math.sin(l)*s+Math.cos(l)*c;u=u*t.width+t.xCenter,d=d*t.height+t.yCenter;var f=o.z*t.width,p=jx({},o);p.x=u,p.y=d,p.z=f,r.push(p)}return r}(h,e),_=function(e,t){for(var n=[],r=0,i=e;r<i.length;r++){var a=i[r],o=a.x,s=a.y,c=t.rotation,l=Math.cos(c)*o-Math.sin(c)*s,u=Math.sin(c)*o+Math.cos(c)*s,d=jx({},a);d.x=l,d.y=u,n.push(d)}return n}(m,e),F(a),F(r),[2,{landmarks:g,worldLandmarks:_,handScore:u,handedness:f}]}}))}))},e.prototype.handLandmarksToRoi=function(e,t){return oS(sS([].concat(e.slice(0,4),e.slice(5,7),e.slice(9,11),e.slice(13,15),e.slice(17,19)),t),t,mS)},e}();function xS(e){return Mx(this,void 0,void 0,(function(){var t,n,r,i,a,o;return Nx(this,(function(s){switch(s.label){case 0:return t=function(e){if(e==null)return jx({},lS);var t=jx({},e);if(t.runtime=`tfjs`,t.maxHands??=lS.maxHands,t.modelType??=lS.modelType,t.modelType!==`lite`&&t.modelType!==`full`)throw Error(`Model type must be one of lite or full, but got `+t.modelType);return t.detectorModelUrl??(t.modelType===`lite`?t.detectorModelUrl=`https://tfhub.dev/mediapipe/tfjs-model/handpose_3d/detector/lite/1`:t.detectorModelUrl=`https://tfhub.dev/mediapipe/tfjs-model/handpose_3d/detector/full/1`),t.landmarkModelUrl??(t.modelType===`lite`?t.landmarkModelUrl=`https://tfhub.dev/mediapipe/tfjs-model/handpose_3d/landmark/lite/1`:t.landmarkModelUrl=`https://tfhub.dev/mediapipe/tfjs-model/handpose_3d/landmark/full/1`),t}(e),n=t.detectorModelUrl.indexOf(`https://tfhub.dev`)>-1,r=t.landmarkModelUrl.indexOf(`https://tfhub.dev`)>-1,[4,Promise.all([kx(t.detectorModelUrl,{fromTFHub:n}),kx(t.landmarkModelUrl,{fromTFHub:r})])];case 1:return i=s.sent(),a=i[0],o=i[1],[2,new bS(a,o,t.maxHands)]}}))}))}function SS(e,t){return Mx(this,void 0,void 0,(function(){var n,r;return Nx(this,(function(i){if(e===yS.MediaPipeHands){if(r=void 0,(n=t)!=null){if(n.runtime===`tfjs`)return[2,xS(n)];if(n.runtime===`mediapipe`)return[2,Lx(n)];r=n.runtime}throw Error(`Expect modelConfig.runtime to be either 'tfjs' or 'mediapipe', but got `+r)}throw Error(e+` is not a supported model name.`)}))}))}(function(e){e.MediaPipeHands=`MediaPipeHands`})(yS||={});var CS={},wS={alpha:!1,antialias:!1,premultipliedAlpha:!1,preserveDrawingBuffer:!1,depth:!1,stencil:!1,failIfMajorPerformanceCaveat:!0};function TS(e,t){CS[e]=t}function ES(e,t){if(!(e in CS)||t!=null){let n=OS(e,t);if(n!==null)CS[e]=n;else return console.log(`Could not get context for WebGL version`,e),null}let n=CS[e];return n==null||n.isContextLost()?(delete CS[e],ES(e)):(n.disable(n.DEPTH_TEST),n.disable(n.STENCIL_TEST),n.disable(n.BLEND),n.disable(n.DITHER),n.disable(n.POLYGON_OFFSET_FILL),n.disable(n.SAMPLE_COVERAGE),n.enable(n.SCISSOR_TEST),n.enable(n.CULL_FACE),n.cullFace(n.BACK),CS[e])}function DS(e){if(!O().getBool(`IS_SAFARI`)&&typeof OffscreenCanvas<`u`&&e===2)return new OffscreenCanvas(300,150);if(typeof document<`u`)return document.createElement(`canvas`);throw Error(`Cannot create a canvas in this context`)}function OS(e,t){if(e!==1&&e!==2)throw Error(`Cannot get WebGL rendering context, WebGL is disabled.`);let n=t??DS(e);return n.addEventListener(`webglcontextlost`,t=>{t.preventDefault(),delete CS[e]},!1),O().getBool(`SOFTWARE_WEBGL_ENABLED`)&&(wS.failIfMajorPerformanceCaveat=!1),e===1?n.getContext(`webgl`,wS)||n.getContext(`experimental-webgl`,wS):n.getContext(`webgl2`,wS)}var kS;(function(e){e[e.DENSE=0]=`DENSE`,e[e.SHARED_BATCH=1]=`SHARED_BATCH`})(kS||={});var AS;(function(e){e[e.RENDER=0]=`RENDER`,e[e.UPLOAD=1]=`UPLOAD`,e[e.PIXELS=2]=`PIXELS`,e[e.DOWNLOAD=3]=`DOWNLOAD`})(AS||={});var jS;(function(e){e[e.UNPACKED_FLOAT16=0]=`UNPACKED_FLOAT16`,e[e.UNPACKED_FLOAT32=1]=`UNPACKED_FLOAT32`,e[e.PACKED_4X1_UNSIGNED_BYTE=2]=`PACKED_4X1_UNSIGNED_BYTE`,e[e.PACKED_2X2_FLOAT32=3]=`PACKED_2X2_FLOAT32`,e[e.PACKED_2X2_FLOAT16=4]=`PACKED_2X2_FLOAT16`})(jS||={});function MS(e,t){return[t,e]}function NS(e,t){return e*t}function PS(e){let t=S(e);return T(Math.ceil(t/4))}function FS(e,t){return[Math.max(1,Math.ceil(t/2)),Math.max(1,Math.ceil(e/2))]}function IS(e,t){let[n,r]=FS(e,t);return n*r*4}function LS(e,t){let n=e,r,i,a,o,s,c,l,u,d,f;return O().getNumber(`WEBGL_VERSION`)===2?(r=n.R32F,i=n.R16F,a=n.RGBA16F,o=n.RGBA32F,s=n.RED,l=4,u=1,d=n.HALF_FLOAT,f=n.FLOAT,c=n.RGBA8):(r=e.RGBA,i=e.RGBA,a=e.RGBA,o=n.RGBA,s=e.RGBA,l=4,u=4,d=t==null?null:t.HALF_FLOAT_OES,f=e.FLOAT,c=e.RGBA),{internalFormatFloat:r,internalFormatHalfFloat:i,internalFormatPackedHalfFloat:a,internalFormatPackedFloat:o,textureFormatFloat:s,downloadTextureFormat:c,downloadUnpackNumChannels:l,defaultNumChannels:u,textureTypeHalfFloat:d,textureTypeFloat:f}}function Y(e,t){let n=t();return O().getBool(`DEBUG`)&&RS(e),n}function RS(e){let t=e.getError();if(t!==e.NO_ERROR)throw Error(`WebGL Error: `+HS(e,t))}var zS=5.96e-8,BS=65504;function VS(e){return!!(O().getBool(`WEBGL_RENDER_FLOAT32_ENABLED`)||e===0||zS<Math.abs(e)&&Math.abs(e)<BS)}function HS(e,t){switch(t){case e.NO_ERROR:return`NO_ERROR`;case e.INVALID_ENUM:return`INVALID_ENUM`;case e.INVALID_VALUE:return`INVALID_VALUE`;case e.INVALID_OPERATION:return`INVALID_OPERATION`;case e.INVALID_FRAMEBUFFER_OPERATION:return`INVALID_FRAMEBUFFER_OPERATION`;case e.OUT_OF_MEMORY:return`OUT_OF_MEMORY`;case e.CONTEXT_LOST_WEBGL:return`CONTEXT_LOST_WEBGL`;default:return`Unknown error code ${t}`}}function US(e,t){return dC(e,()=>e.getExtension(t),`Extension "`+t+`" not supported on this browser.`)}function WS(e,t){let n=dC(e,()=>e.createShader(e.VERTEX_SHADER),`Unable to create vertex WebGLShader.`);if(Y(e,()=>e.shaderSource(n,t)),Y(e,()=>e.compileShader(n)),e.getShaderParameter(n,e.COMPILE_STATUS)===!1)throw console.log(e.getShaderInfoLog(n)),Error(`Failed to compile vertex shader.`);return n}function GS(e,t){let n=dC(e,()=>e.createShader(e.FRAGMENT_SHADER),`Unable to create fragment WebGLShader.`);if(Y(e,()=>e.shaderSource(n,t)),Y(e,()=>e.compileShader(n)),O().get(`ENGINE_COMPILE_ONLY`))return n;if(e.getShaderParameter(n,e.COMPILE_STATUS)===!1)throw qS(t,e.getShaderInfoLog(n)),Error(`Failed to compile fragment shader.`);return n}var KS=/ERROR: [0-9]+:([0-9]+):/g;function qS(e,t){let n=KS.exec(t);if(n==null){console.log(`Couldn't parse line number in error: ${t}`),console.log(e);return}let r=+n[1],i=e.split(`
`),a=i.length.toString().length+2,o=i.map((e,t)=>E((t+1).toString(),a)+e),s=0;for(let e=0;e<o.length;e++)s=Math.max(o[e].length,s);let c=o.slice(0,r-1),l=o.slice(r-1,r),u=o.slice(r);console.log(c.join(`
`)),console.log(t.split(`
`)[0]),console.log(`%c ${E(l[0],s)}`,`border:1px solid red; background-color:#e3d2d2; color:#a61717`),console.log(u.join(`
`))}function JS(e){return dC(e,()=>e.createProgram(),`Unable to create WebGLProgram.`)}function YS(e,t){if(Y(e,()=>e.linkProgram(t)),!O().get(`ENGINE_COMPILE_ONLY`)&&e.getProgramParameter(t,e.LINK_STATUS)===!1)throw console.log(e.getProgramInfoLog(t)),Error(`Failed to link vertex and fragment shaders.`)}function XS(e,t){if(Y(e,()=>e.validateProgram(t)),e.getProgramParameter(t,e.VALIDATE_STATUS)===!1)throw console.log(e.getProgramInfoLog(t)),Error(`Shader program validation failed.`)}function ZS(e,t){let n=dC(e,()=>e.createBuffer(),`Unable to create WebGLBuffer`);return Y(e,()=>e.bindBuffer(e.ARRAY_BUFFER,n)),Y(e,()=>e.bufferData(e.ARRAY_BUFFER,t,e.STATIC_DRAW)),n}function QS(e,t){let n=dC(e,()=>e.createBuffer(),`Unable to create WebGLBuffer`);return Y(e,()=>e.bindBuffer(e.ELEMENT_ARRAY_BUFFER,n)),Y(e,()=>e.bufferData(e.ELEMENT_ARRAY_BUFFER,t,e.STATIC_DRAW)),n}function $S(e){return dC(e,()=>e.createTexture(),`Unable to create WebGLTexture.`)}function eC(e,t){let n=O().getNumber(`WEBGL_MAX_TEXTURE_SIZE`);if(e<=0||t<=0){let n=`[${e}x${t}]`;throw Error(`Requested texture size `+n+` is invalid.`)}if(e>n||t>n){let r=`[${e}x${t}]`,i=`[${n}x${n}]`;throw Error(`Requested texture size `+r+` greater than WebGL maximum on this browser / GPU `+i+`.`)}}function tC(e){return dC(e,()=>e.createFramebuffer(),`Unable to create WebGLFramebuffer.`)}function nC(e,t,n,r,i,a,o){let s=e.getAttribLocation(t,n);return s===-1?!1:(Y(e,()=>e.bindBuffer(e.ARRAY_BUFFER,r)),Y(e,()=>e.vertexAttribPointer(s,i,e.FLOAT,!1,a,o)),Y(e,()=>e.enableVertexAttribArray(s)),!0)}function rC(e,t,n){fC(e,n),Y(e,()=>e.activeTexture(e.TEXTURE0+n)),Y(e,()=>e.bindTexture(e.TEXTURE_2D,t))}function iC(e,t,n){return dC(e,()=>e.getUniformLocation(t,n),`uniform "`+n+`" not present in program.`)}function aC(e,t,n){return e.getUniformLocation(t,n)}function oC(e,t,n,r){Y(e,()=>rC(e,t,r)),Y(e,()=>e.uniform1i(n,r))}function sC(e,t,n){Y(e,()=>e.bindFramebuffer(e.FRAMEBUFFER,n)),Y(e,()=>e.framebufferTexture2D(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,t,0))}function cC(e,t){Y(e,()=>e.bindFramebuffer(e.FRAMEBUFFER,t)),Y(e,()=>e.framebufferTexture2D(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,null,0))}function lC(e){let t=e.checkFramebufferStatus(e.FRAMEBUFFER);if(t!==e.FRAMEBUFFER_COMPLETE)throw Error(`Error binding framebuffer: `+uC(e,t))}function uC(e,t){switch(t){case e.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:return`FRAMEBUFFER_INCOMPLETE_ATTACHMENT`;case e.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:return`FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT`;case e.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:return`FRAMEBUFFER_INCOMPLETE_DIMENSIONS`;case e.FRAMEBUFFER_UNSUPPORTED:return`FRAMEBUFFER_UNSUPPORTED`;default:return`unknown error ${t}`}}function dC(e,t,n){let r=Y(e,()=>t());if(r==null)throw Error(n);return r}function fC(e,t){let n=e.MAX_COMBINED_TEXTURE_IMAGE_UNITS-1,r=t+e.TEXTURE0;if(r<e.TEXTURE0||r>n){let e=`[gl.TEXTURE0, gl.TEXTURE${n}]`;throw Error(`textureUnit must be in ${e}.`)}}function pC(e,t=2){return S(e.slice(0,e.length-t))}function mC(e){if(e.length===0)throw Error(`Cannot get rows and columns of an empty shape array.`);return[e.length>1?e[e.length-2]:1,e[e.length-1]]}function hC(e){let t=[1,1,1];return e.length===0||e.length===1&&e[0]===1||(t=[pC(e),...mC(e)]),t}function gC(e,t=!1){let n=O().getNumber(`WEBGL_MAX_TEXTURE_SIZE`),r=O().getNumber(`WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE`);r===1/0&&O().getBool(`WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE`)&&(r=n/2),t&&(n*=2,r*=2,e=e.map((t,n)=>n>=e.length-2?g(e[n]):e[n]),e.length===1&&(e=[2,e[0]])),e.length!==2&&(e=re(e).newShape);let i=S(e),a=null;e.length<=1&&i<=n?a=[1,i]:e.length===2&&e[0]<=n&&e[1]<=n?a=e:e.length===3&&e[0]*e[1]<=n&&e[2]<=n?a=[e[0]*e[1],e[2]]:e.length===3&&e[0]<=n&&e[1]*e[2]<=n?a=[e[0],e[1]*e[2]]:e.length===4&&e[0]*e[1]*e[2]<=n&&e[3]<=n?a=[e[0]*e[1]*e[2],e[3]]:e.length===4&&e[0]<=n&&e[1]*e[2]*e[3]<=n&&(a=[e[0],e[1]*e[2]*e[3]]);let o=a!=null&&Math.max(...a)>r&&Math.min(...a)<=(t?2:1)&&Math.min(...a)>0;if(a==null||o)if(t){let t=pC(e),n=2,r=2;e.length&&([n,r]=mC(e)),i=n/2*t*(r/2),a=T(i).map(e=>e*2)}else a=T(i);return a}function _C(e){return e%2==0}function vC(e,t){if(e=e.slice(-2),t=t.slice(-2),w(e,t)||!e.length||!t.length||e[0]===0||e[1]===0||t[0]===0||t[1]===0)return!0;if(e.length!==t.length){let n=e[e.length-1],r=t[t.length-1];if(n===r||_C(n)&&_C(r)&&(e[0]===1||t[0]===1))return!0}return e[1]===t[1]&&_C(e[0])&&_C(t[0])}var yC,bC;function xC(e){if(yC==null){let t=ES(e);yC=t.getParameter(t.MAX_TEXTURE_SIZE)}return yC}function SC(e){if(bC==null){let t=ES(e);bC=t.getParameter(t.MAX_TEXTURE_IMAGE_UNITS)}return Math.min(16,bC)}function CC(e){if(e===0)return 0;let t,n=ES(e);return t=wC(n,`EXT_disjoint_timer_query_webgl2`)&&e===2?2:+!!wC(n,`EXT_disjoint_timer_query`),t}function wC(e,t){return e.getExtension(t)!=null}function TC(e){try{if(ES(e)!=null)return!0}catch(e){return console.log(`Error when getting WebGL context: `,e),!1}return!1}function EC(e){if(e===0)return!1;let t=ES(e);if(e===1){if(!wC(t,`OES_texture_float`))return!1}else if(!wC(t,`EXT_color_buffer_float`))return!1;return OC(t)}function DC(e){if(e===0)return!1;let t=ES(e);if(e===1){if(!wC(t,`OES_texture_float`)||!wC(t,`WEBGL_color_buffer_float`))return!1}else{if(wC(t,`EXT_color_buffer_float`))return OC(t);let e=`EXT_color_buffer_half_float`;return wC(t,e)?kC(t,t.getExtension(e)):!1}return OC(t)}function OC(e){let t=LS(e),n=e.createTexture();e.bindTexture(e.TEXTURE_2D,n),e.texImage2D(e.TEXTURE_2D,0,t.internalFormatFloat,1,1,0,t.textureFormatFloat,t.textureTypeFloat,null);let r=e.createFramebuffer();e.bindFramebuffer(e.FRAMEBUFFER,r),e.framebufferTexture2D(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,n,0);let i=e.checkFramebufferStatus(e.FRAMEBUFFER)===e.FRAMEBUFFER_COMPLETE;return e.bindTexture(e.TEXTURE_2D,null),e.bindFramebuffer(e.FRAMEBUFFER,null),e.deleteTexture(n),e.deleteFramebuffer(r),i}function kC(e,t){let n=LS(e,t),r=e.createTexture();e.bindTexture(e.TEXTURE_2D,r),e.texImage2D(e.TEXTURE_2D,0,n.internalFormatHalfFloat,1,1,0,n.textureFormatFloat,n.textureTypeHalfFloat,null);let i=e.createFramebuffer();e.bindFramebuffer(e.FRAMEBUFFER,i),e.framebufferTexture2D(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,r,0);let a=e.checkFramebufferStatus(e.FRAMEBUFFER)===e.FRAMEBUFFER_COMPLETE;return e.bindTexture(e.TEXTURE_2D,null),e.bindFramebuffer(e.FRAMEBUFFER,null),e.deleteTexture(r),e.deleteFramebuffer(i),a}function AC(e){return e===2?ES(e).fenceSync!=null:!1}function jC(e,t){Array.isArray(e)||(e=[e]),e.forEach(e=>{e!=null&&y(e.dtype!==`complex64`,()=>`${t} does not support complex64 tensors in the WebGL backend.`)})}var X=O();X.registerFlag(`HAS_WEBGL`,()=>X.getNumber(`WEBGL_VERSION`)>0),X.registerFlag(`WEBGL_VERSION`,()=>TC(2)?2:+!!TC(1)),X.registerFlag(`WEBGL_CHECK_NUMERICAL_PROBLEMS`,()=>!1),X.registerFlag(`WEBGL_BUFFER_SUPPORTED`,()=>X.get(`WEBGL_VERSION`)===2),X.registerFlag(`WEBGL_CPU_FORWARD`,()=>!0),X.registerFlag(`WEBGL_FORCE_F16_TEXTURES`,()=>!1),X.registerFlag(`WEBGL_PACK`,()=>X.getBool(`HAS_WEBGL`)),X.registerFlag(`WEBGL_PACK_NORMALIZATION`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_PACK_CLIP`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_PACK_DEPTHWISECONV`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_PACK_BINARY_OPERATIONS`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_PACK_UNARY_OPERATIONS`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_PACK_ARRAY_OPERATIONS`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_PACK_IMAGE_OPERATIONS`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_PACK_REDUCE`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_LAZILY_UNPACK`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_CONV_IM2COL`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_PACK_CONV2DTRANSPOSE`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_MAX_TEXTURE_SIZE`,()=>xC(X.getNumber(`WEBGL_VERSION`))),X.registerFlag(`WEBGL_MAX_TEXTURES_IN_SHADER`,()=>SC(X.getNumber(`WEBGL_VERSION`))),X.registerFlag(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`,()=>{let e=X.getNumber(`WEBGL_VERSION`);return e===0?0:CC(e)}),X.registerFlag(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE`,()=>X.getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`)>0&&!Zi()),X.registerFlag(`WEBGL_RENDER_FLOAT32_CAPABLE`,()=>EC(X.getNumber(`WEBGL_VERSION`))),X.registerFlag(`WEBGL_RENDER_FLOAT32_ENABLED`,()=>X.getBool(`WEBGL_FORCE_F16_TEXTURES`)?!1:X.getBool(`WEBGL_RENDER_FLOAT32_CAPABLE`)),X.registerFlag(`WEBGL_DOWNLOAD_FLOAT_ENABLED`,()=>DC(X.getNumber(`WEBGL_VERSION`))),X.registerFlag(`WEBGL_FENCE_API_ENABLED`,()=>AC(X.getNumber(`WEBGL_VERSION`))),X.registerFlag(`WEBGL_SIZE_UPLOAD_UNIFORM`,()=>X.getBool(`WEBGL_RENDER_FLOAT32_ENABLED`)?4:0),X.registerFlag(`WEBGL_DELETE_TEXTURE_THRESHOLD`,()=>-1,e=>{if(typeof e!=`number`)throw Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be a number but got ${e}.`);if(e<0&&e!==-1)throw Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be -1 (indicating never delete) or at least 0, but got ${e}.`)}),X.registerFlag(`WEBGL_FLUSH_THRESHOLD`,()=>Zi()?1:-1,e=>{if(typeof e!=`number`)throw Error(`WEBGL_FLUSH_THRESHOLD must be a number but got ${e}.`);if(e<0&&e!==-1)throw Error(`WEBGL_FLUSH_THRESHOLD must be -1 (indicating never manual flush) or at least 0, but got ${e}.`)}),X.registerFlag(`CPU_HANDOFF_SIZE_THRESHOLD`,()=>128),X.registerFlag(`WEBGL_USE_SHAPES_UNIFORMS`,()=>!1),X.registerFlag(`TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD`,()=>1e5),X.registerFlag(`TOPK_K_CPU_HANDOFF_THRESHOLD`,()=>128),X.registerFlag(`WEBGL_EXP_CONV`,()=>!1),X.registerFlag(`SOFTWARE_WEBGL_ENABLED`,()=>X.getBool(`IS_TEST`)),X.registerFlag(`WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE`,()=>1/0),X.registerFlag(`WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE`,()=>!1),X.registerFlag(`WEBGL2_ISNAN_CUSTOM`,()=>!1),X.registerFlag(`ENGINE_COMPILE_ONLY`,()=>!1);function MC(){let e,t,n,r,i,a,o,s,c,l;return O().getNumber(`WEBGL_VERSION`)===2?(e=`#version 300 es`,t=`in`,n=`out`,r=`in`,i=`texture`,a=`outputColor`,o=`out vec4 outputColor;`,s=O().getBool(`WEBGL2_ISNAN_CUSTOM`)?`
      bool isnan_custom(float val) {
        uint floatToUint = floatBitsToUint(val);
        return (floatToUint & 0x7fffffffu) > 0x7f800000u;
      }

      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan_custom(val.x),
          isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));
      }

      #define isnan(value) isnan_custom(value)
    `:``,c=``,l=`
      #define round(value) newRound(value)
      int newRound(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 newRound(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `):(e=``,t=`attribute`,n=`varying`,r=`varying`,i=`texture2D`,a=`gl_FragColor`,o=``,s=`
      #define isnan(value) isnan_custom(value)
      bool isnan_custom(float val) {
        return (val > 0. || val < 1. || val == 0.) ? false : true;
      }
      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan(val.x), isnan(val.y), isnan(val.z), isnan(val.w));
      }
    `,c=`
      uniform float INFINITY;

      bool isinf(float val) {
        return abs(val) == INFINITY;
      }
      bvec4 isinf(vec4 val) {
        return equal(abs(val), vec4(INFINITY));
      }
    `,l=`
      int round(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 round(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `),{version:e,attribute:t,varyingVs:n,varyingFs:r,texture2D:i,output:a,defineOutput:o,defineSpecialNaN:s,defineSpecialInf:c,defineRound:l}}function NC(e,t,n=`index`){let r=_e(t);return r.map((t,i)=>`${`int ${e[i]} = ${n} / ${t}`}; ${i===r.length-1?`int ${e[i+1]} = ${n} - ${e[i]} * ${t}`:`index -= ${e[i]} * ${t}`};`).join(``)}function PC(e,t,n=`index`){let r=_e(t);return r.map((t,i)=>`${`int ${e[i]} = ${n} / outShapeStrides[${i}]`}; ${i===r.length-1?`int ${e[i+1]} = ${n} - ${e[i]} * outShapeStrides[${i}]`:`index -= ${e[i]} * outShapeStrides[${i}]`};`).join(``)}function FC(e,t){let n=e.length,r=e.map(e=>`${t}[${e}]`),i=Array(n-1);i[n-2]=r[n-1];for(let e=n-3;e>=0;--e)i[e]=`(${i[e+1]} * ${r[e+1]})`;return i}function IC(e,t,n=`index`){let r=FC(e.map((e,t)=>t),t);return r.map((t,i)=>`${`int ${e[i]} = ${n} / ${r[i]}`}; ${i===r.length-1?`int ${e[i+1]} = ${n} - ${e[i]} * ${r[i]}`:`index -= ${e[i]} * ${r[i]}`};`).join(``)}function LC(e){let t=_e(e).map(e=>e.toString());return`
  int getFlatIndex(ivec3 coords) {
    return coords.x * ${t[0]} + coords.y * ${t[1]} + coords.z;
  }
`}function RC(){return`
  int getFlatIndex(ivec3 coords) {
    return coords.x * outShapeStrides[0] + coords.y * outShapeStrides[1] + coords.z;
  }
`}var zC=`
  const float FLOAT_MAX = 1.70141184e38;
  const float FLOAT_MIN = 1.17549435e-38;

  lowp vec4 encode_float(highp float v) {
    if (isnan(v)) {
      return vec4(255, 255, 255, 255);
    }

    highp float av = abs(v);

    if(av < FLOAT_MIN) {
      return vec4(0.0, 0.0, 0.0, 0.0);
    } else if(v > FLOAT_MAX) {
      return vec4(0.0, 0.0, 128.0, 127.0) / 255.0;
    } else if(v < -FLOAT_MAX) {
      return vec4(0.0, 0.0,  128.0, 255.0) / 255.0;
    }

    highp vec4 c = vec4(0,0,0,0);

    highp float e = floor(log2(av));
    highp float m = exp2(fract(log2(av))) - 1.0;

    c[2] = floor(128.0 * m);
    m -= c[2] / 128.0;
    c[1] = floor(32768.0 * m);
    m -= c[1] / 32768.0;
    c[0] = floor(8388608.0 * m);

    highp float ebias = e + 127.0;
    c[3] = floor(ebias / 2.0);
    ebias -= c[3] * 2.0;
    c[2] += floor(ebias) * 128.0;

    c[3] += 128.0 * step(0.0, -v);

    return c / 255.0;
  }
`,{getBroadcastDims:BC}=hy;function VC(e,t,n){let r=[];if(e.forEach(e=>{let t=S(e.shapeInfo.logicalShape);if(e.shapeInfo.isUniform?r.push(`uniform float ${e.name}${t>1?`[${t}]`:``};`):(r.push(`uniform sampler2D ${e.name};`),r.push(`uniform int offset${e.name};`)),n.enableShapeUniforms){let{uniformShape:t}=Ow(n.packedInputs,e.shapeInfo.logicalShape,e.shapeInfo.texShape);switch(t.length){case 1:r.push(`uniform int ${e.name}Shape;`);break;case 2:r.push(`uniform ivec2 ${e.name}Shape;`);break;case 3:r.push(`uniform ivec3 ${e.name}Shape;`);break;case 4:r.push(`uniform ivec4 ${e.name}Shape;`);break;default:break}r.push(`uniform ivec2 ${e.name}TexShape;`)}}),n.enableShapeUniforms){switch(t.logicalShape.length){case 1:r.push(`uniform int outShape;`);break;case 2:r.push(`uniform ivec2 outShape;`),r.push(`uniform int outShapeStrides;`);break;case 3:r.push(`uniform ivec3 outShape;`),r.push(`uniform ivec2 outShapeStrides;`);break;case 4:r.push(`uniform ivec4 outShape;`),r.push(`uniform ivec3 outShapeStrides;`);break;default:break}r.push(`uniform ivec2 outTexShape;`)}n.customUniforms&&n.customUniforms.forEach(e=>{r.push(`uniform ${e.type} ${e.name}${e.arrayIndex?`[${e.arrayIndex}]`:``};`)});let i=r.join(`
`),a=e.map(e=>WC(e,t,n.packedInputs,n.enableShapeUniforms)).join(`
`),o=t.texShape,s=MC(),c=qC(s),l,u,d=XC(s);return t.isPacked?(l=GC(t.logicalShape,o,n.enableShapeUniforms),u=YC(s)):(l=KC(t.logicalShape,o,n.enableShapeUniforms),u=JC(s)),n.packedInputs&&(d+=ew),[d,c,u,i,l,a,n.userCode].join(`
`)}function HC(e,t=!1){let n=e.shapeInfo.logicalShape;switch(n.length){case 0:return mw(e,t);case 1:return gw(e,t);case 2:return vw(e,t);case 3:return bw(e,t);case 4:return Sw(e,t);case 5:return Cw(e);case 6:return ww(e);default:throw Error(`${n.length}-D input sampling is not yet supported`)}}function UC(e,t){switch(e.shapeInfo.logicalShape.length){case 0:return pw(e);case 1:return hw(e,t);case 2:return _w(e,t);case 3:return yw(e,t);default:return xw(e,t)}}function WC(e,t,n=!1,r){let i=``;n?i+=UC(e,r):i+=HC(e,r);let a=e.shapeInfo.logicalShape,o=t.logicalShape;return a.length<=o.length&&(n?i+=Ew(e,t):i+=Dw(e,t)),i}function GC(e,t,n){switch(e.length){case 0:return tw();case 1:return nw(e,t,n);case 2:return uw(e,t,n);case 3:return iw(e,t,n);default:return ow(e,t,n)}}function KC(e,t,n){switch(e.length){case 0:return tw();case 1:return rw(e,t,n);case 2:return dw(e,t,n);case 3:return aw(e,t,n);case 4:return sw(e,t,n);case 5:return cw(e,t);case 6:return lw(e,t);default:throw Error(`${e.length}-D output sampling is not yet supported`)}}function qC(e){return`
    float sampleTexture(sampler2D textureSampler, vec2 uv) {
      return ${e.texture2D}(textureSampler, uv).r;
    }
  `}function JC(e){return`
    void setOutput(float val) {
      ${e.output} = vec4(val, 0, 0, 0);
    }
  `}function YC(e){return`
    void setOutput(vec4 val) {
      ${e.output} = val;
    }
  `}function XC(e){return`${e.version}
    precision highp float;
    precision highp int;
    precision highp sampler2D;
    ${e.varyingFs} vec2 resultUV;
    ${e.defineOutput}
    const vec2 halfCR = vec2(0.5, 0.5);

    struct ivec5
    {
      int x;
      int y;
      int z;
      int w;
      int u;
    };

    struct ivec6
    {
      int x;
      int y;
      int z;
      int w;
      int u;
      int v;
    };

    uniform float NAN;
    ${e.defineSpecialNaN}
    ${e.defineSpecialInf}
    ${e.defineRound}

    int imod(int x, int y) {
      return x - y * (x / y);
    }

    int idiv(int a, int b, float sign) {
      int res = a / b;
      int mod = imod(a, b);
      if (sign < 0. && mod != 0) {
        res -= 1;
      }
      return res;
    }

    //Based on the work of Dave Hoskins
    //https://www.shadertoy.com/view/4djSRW
    #define HASHSCALE1 443.8975
    float random(float seed){
      vec2 p = resultUV * seed;
      vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);
      p3 += dot(p3, p3.yzx + 19.19);
      return fract((p3.x + p3.y) * p3.z);
    }

    ${ZC}
    ${QC}
    ${$C}
  `}var ZC=`
vec2 uvFromFlat(int texNumR, int texNumC, int index) {
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
vec2 packedUVfrom1D(int texNumR, int texNumC, int index) {
  int texelIndex = index / 2;
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,QC=`
vec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,
  int texNumC, int row, int col) {
  int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,$C=`
vec2 packedUVfrom3D(int texNumR, int texNumC,
    int texelsInBatch, int texelsInLogicalRow, int b,
    int row, int col) {
  int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,ew=`
  float getChannel(vec4 frag, vec2 innerDims) {
    vec2 modCoord = mod(innerDims, 2.);
    return modCoord.x == 0. ?
      (modCoord.y == 0. ? frag.r : frag.g) :
      (modCoord.y == 0. ? frag.b : frag.a);
  }
  float getChannel(vec4 frag, int dim) {
    float modCoord = mod(float(dim), 2.);
    return modCoord == 0. ? frag.r : frag.g;
  }
`;function tw(){return`
    int getOutputCoords() {
      return 0;
    }
  `}function nw(e,t,n){let r=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)];return r[0]===1?n?`
      int getOutputCoords() {
        return 2 * int(resultUV.x * ceil(float(outTexShape[1]) / 2.0));
      }
    `:`
      int getOutputCoords() {
        return 2 * int(resultUV.x * ${r[1]}.0);
      }
    `:r[1]===1?n?`
      int getOutputCoords() {
        return 2 * int(resultUV.y * ceil(float(outTexShape[0]) / 2.0));
      }
    `:`
      int getOutputCoords() {
        return 2 * int(resultUV.y * ${r[0]}.0);
      }
    `:n?`
    int getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      return 2 * (resTexRC.x * packedTexShape[1] + resTexRC.y);
    }
  `:`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${r[0]}, ${r[1]}));
      return 2 * (resTexRC.x * ${r[1]} + resTexRC.y);
    }
  `}function rw(e,t,n){return t[0]===1?n?`
      int getOutputCoords() {
        return int(resultUV.x * float(outTexShape[1]));
      }
    `:`
      int getOutputCoords() {
        return int(resultUV.x * ${t[1]}.0);
      }
    `:t[1]===1?n?`
      int getOutputCoords() {
        return int(resultUV.y * float(outTexShape[0]));
      }
    `:`
      int getOutputCoords() {
        return int(resultUV.y * ${t[0]}.0);
      }
    `:n?`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      return resTexRC.x * outTexShape[1] + resTexRC.y;
    }
  `:`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${t[0]}, ${t[1]}));
      return resTexRC.x * ${t[1]} + resTexRC.y;
    }
  `}function iw(e,t,n){if(n)return`
    ivec3 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec3(b, r, c);
    }
  `;let r=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)],i=Math.ceil(e[2]/2),a=i*Math.ceil(e[1]/2);return`
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${r[0]}, ${r[1]}));
      int index = resTexRC.x * ${r[1]} + resTexRC.y;

      int b = index / ${a};
      index -= b * ${a};

      int r = 2 * (index / ${i});
      int c = imod(index, ${i}) * 2;

      return ivec3(b, r, c);
    }
  `}function aw(e,t,n){if(n)return`
  ivec3 getOutputCoords() {
    ivec2 resTexRC = ivec2(resultUV.yx *
                           vec2(outTexShape[0], outTexShape[1]));
    int index = resTexRC.x * outTexShape[1] + resTexRC.y;
    ${PC([`r`,`c`,`d`],e)}
    return ivec3(r, c, d);
  }
`;let r=NC([`r`,`c`,`d`],e);return`
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;
      ${r}
      return ivec3(r, c, d);
    }
  `}function ow(e,t,n){if(n)return`
    ivec4 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int texelsInLogicalRow = int(ceil(float(outShape[3]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatchN = texelsInBatch * outShape[1];

      int b2 = index / texelsInBatchN;
      index -= b2 * texelsInBatchN;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec4(b2, b, r, c);
    }
  `;let r=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)],i=Math.ceil(e[e.length-1]/2),a=i*Math.ceil(e[e.length-2]/2),o=a,s=``,c=`b, r, c`;for(let t=2;t<e.length-1;t++)o*=e[e.length-t-1],s=`
      int b${t} = index / ${o};
      index -= b${t} * ${o};
    `+s,c=`b${t}, `+c;return`
    ivec${e.length} getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${r[0]}, ${r[1]}));
      int index = resTexRC.x * ${r[1]} + resTexRC.y;

      ${s}

      int b = index / ${a};
      index -= b * ${a};

      int r = 2 * (index / ${i});
      int c = imod(index, ${i}) * 2;

      return ivec${e.length}(${c});
    }
  `}function sw(e,t,n){if(n)return`
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      ${PC([`r`,`c`,`d`,`d2`],e)}
      return ivec4(r, c, d, d2);
    }
  `;let r=NC([`r`,`c`,`d`,`d2`],e);return`
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;
      ${r}
      return ivec4(r, c, d, d2);
    }
  `}function cw(e,t){let n=NC([`r`,`c`,`d`,`d2`,`d3`],e);return`
    ivec5 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx * vec2(${t[0]},
                             ${t[1]}));

      int index = resTexRC.x * ${t[1]} + resTexRC.y;

      ${n}

      ivec5 outShape = ivec5(r, c, d, d2, d3);
      return outShape;
    }
  `}function lw(e,t){let n=NC([`r`,`c`,`d`,`d2`,`d3`,`d4`],e);return`
    ivec6 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;

      ${n}

      ivec6 result = ivec6(r, c, d, d2, d3, d4);
      return result;
    }
  `}function uw(e,t,n){let r=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)];if(w(e,t))return n?`
      ivec2 getOutputCoords() {
        ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
        return 2 * ivec2(resultUV.yx * vec2(packedTexShape[0], packedTexShape[1]));
      }
    `:`
      ivec2 getOutputCoords() {
        return 2 * ivec2(resultUV.yx * vec2(${r[0]}, ${r[1]}));
      }
    `;let i=Math.ceil(e[1]/2);return n?`
    ivec2 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));

      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;
      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec2(r, c);
    }
  `:`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${r[0]}, ${r[1]}));

      int index = resTexRC.x * ${r[1]} + resTexRC.y;
      int r = 2 * (index / ${i});
      int c = imod(index, ${i}) * 2;

      return ivec2(r, c);
    }
  `}function dw(e,t,n){return w(e,t)?n?`
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(outTexShape[0], outTexShape[1]));
      }
    `:`
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(${t[0]}, ${t[1]}));
      }
    `:e[1]===1?n?`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(index, 0);
      }
    `:`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${t[0]}, ${t[1]}));
        int index = resTexRC.x * ${t[1]} + resTexRC.y;
        return ivec2(index, 0);
      }
    `:e[0]===1?n?`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(0, index);
      }
    `:`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${t[0]}, ${t[1]}));
        int index = resTexRC.x * ${t[1]} + resTexRC.y;
        return ivec2(0, index);
      }
    `:n?`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      int r = index / outShape[1];
      int c = index - r * outShape[1];
      return ivec2(r, c);
    }
  `:`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;
      int r = index / ${e[1]};
      int c = index - r * ${e[1]};
      return ivec2(r, c);
    }
  `}function fw(e){return`offset${e}`}function pw(e){let t=e.name;return`
    vec4 ${`get`+t.charAt(0).toUpperCase()+t.slice(1)}() {
      return ${MC().texture2D}(${t}, halfCR);
    }
  `}function mw(e,t){let n=e.name,r=`get`+n.charAt(0).toUpperCase()+n.slice(1);if(e.shapeInfo.isUniform)return`float ${r}() {return ${n};}`;let[i,a]=e.shapeInfo.texShape;if(i===1&&a===1)return`
      float ${r}() {
        return sampleTexture(${n}, halfCR);
      }
    `;let o=fw(n);if(t)return`
    float ${r}() {
      vec2 uv = uvFromFlat(${n}TexShape[0], ${n}TexShape[1], ${o});
      return sampleTexture(${n}, uv);
    }
  `;let[s,c]=e.shapeInfo.texShape;return`
    float ${r}() {
      vec2 uv = uvFromFlat(${s}, ${c}, ${o});
      return sampleTexture(${n}, uv);
    }
  `}function hw(e,t){let n=e.name,r=`get`+n.charAt(0).toUpperCase()+n.slice(1),i=e.shapeInfo.texShape,a=MC();if(t)return`
    vec4 ${r}(int index) {
      ivec2 packedTexShape = ivec2(ceil(float(${n}TexShape[0]) / 2.0), ceil(float(${n}TexShape[1]) / 2.0));
      vec2 uv = packedUVfrom1D(
        packedTexShape[0], packedTexShape[1], index);
      return ${a.texture2D}(${n}, uv);
    }
  `;let o=[Math.ceil(i[0]/2),Math.ceil(i[1]/2)];return`
    vec4 ${r}(int index) {
      vec2 uv = packedUVfrom1D(
        ${o[0]}, ${o[1]}, index);
      return ${a.texture2D}(${n}, uv);
    }
  `}function gw(e,t){let n=e.name,r=`get`+n.charAt(0).toUpperCase()+n.slice(1);if(e.shapeInfo.isUniform)return`
      float ${r}(int index) {
        ${Tw(e)}
      }
    `;let i=e.shapeInfo.texShape,a=i[0],o=i[1];if(o===1&&a===1)return`
      float ${r}(int index) {
        return sampleTexture(${n}, halfCR);
      }
    `;let s=fw(n);return o===1?t?`
      float ${r}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${s}) + 0.5) / float(${n}TexShape[0]));
        return sampleTexture(${n}, uv);
      }
    `:`
      float ${r}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${s}) + 0.5) / ${a}.0);
        return sampleTexture(${n}, uv);
      }
    `:a===1?t?`
      float ${r}(int index) {
        vec2 uv = vec2((float(index + ${s}) + 0.5) / float(${n}TexShape[1]), 0.5);
        return sampleTexture(${n}, uv);
      }
    `:`
      float ${r}(int index) {
        vec2 uv = vec2((float(index + ${s}) + 0.5) / ${o}.0, 0.5);
        return sampleTexture(${n}, uv);
      }
    `:t?`
    float ${r}(int index) {
      vec2 uv = uvFromFlat(${n}TexShape[0], ${n}TexShape[1], index + ${s});
      return sampleTexture(${n}, uv);
    }
  `:`
    float ${r}(int index) {
      vec2 uv = uvFromFlat(${a}, ${o}, index + ${s});
      return sampleTexture(${n}, uv);
    }
  `}function _w(e,t){let n=e.shapeInfo.logicalShape,r=e.name,i=`get`+r.charAt(0).toUpperCase()+r.slice(1),a=e.shapeInfo.texShape,o=a[0],s=a[1],c=MC();if(a!=null&&w(n,a))return t?`
      vec4 ${i}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${r}TexShape[1], ${r}TexShape[0]);

        return ${c.texture2D}(${r}, uv);
      }
    `:`
      vec4 ${i}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${s}.0, ${o}.0);

        return ${c.texture2D}(${r}, uv);
      }
    `;if(t)return`
    vec4 ${i}(int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${r}TexShape[0]) / 2.0), ceil(float(${r}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${r}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom2D(valuesPerRow, packedTexShape[0], packedTexShape[1], row, col);
      return ${c.texture2D}(${r}, uv);
    }
  `;let l=[Math.ceil(a[0]/2),Math.ceil(a[1]/2)];return`
    vec4 ${i}(int row, int col) {
      vec2 uv = packedUVfrom2D(${Math.ceil(n[1]/2)}, ${l[0]}, ${l[1]}, row, col);
      return ${c.texture2D}(${r}, uv);
    }
  `}function vw(e,t){let n=e.shapeInfo.logicalShape,r=e.name,i=`get`+r.charAt(0).toUpperCase()+r.slice(1),a=e.shapeInfo.texShape;if(a!=null&&w(n,a)){if(t)return`
      float ${i}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${r}TexShape[1], ${r}TexShape[0]);
        return sampleTexture(${r}, uv);
      }
    `;let e=a[0];return`
    float ${i}(int row, int col) {
      vec2 uv = (vec2(col, row) + halfCR) / vec2(${a[1]}.0, ${e}.0);
      return sampleTexture(${r}, uv);
    }
  `}let{newShape:o,keptDims:s}=re(n),c=o;if(c.length<n.length)return`
      ${HC(kw(e,c),t)}
      float ${i}(int row, int col) {
        return ${i}(${Aw([`row`,`col`],s)});
      }
    `;if(e.shapeInfo.isUniform)return`
      float ${i}(int row, int col) {
        int index = round(dot(vec2(row, col), vec2(${n[1]}, 1)));
        ${Tw(e)}
      }
    `;let l=a[0],u=a[1],d=fw(r);return u===1?t?`
      float ${i}(int row, int col) {
        float index = dot(vec3(row, col, ${d}), vec3(${r}Shape[1], 1, 1));
        vec2 uv = vec2(0.5, (index + 0.5) / float(${r}TexShape[0]));
        return sampleTexture(${r}, uv);
      }
    `:`
    float ${i}(int row, int col) {
      float index = dot(vec3(row, col, ${d}), vec3(${n[1]}, 1, 1));
      vec2 uv = vec2(0.5, (index + 0.5) / ${l}.0);
      return sampleTexture(${r}, uv);
    }
  `:l===1?t?`
      float ${i}(int row, int col) {
        float index = dot(vec3(row, col, ${d}), vec3(${r}Shape[1], 1, 1));
        vec2 uv = vec2((index + 0.5) / float(${r}TexShape[1]), 0.5);
        return sampleTexture(${r}, uv);
      }
    `:`
    float ${i}(int row, int col) {
      float index = dot(vec3(row, col, ${d}), vec3(${n[1]}, 1, 1));
      vec2 uv = vec2((index + 0.5) / ${u}.0, 0.5);
      return sampleTexture(${r}, uv);
    }
  `:t?`
      float ${i}(int row, int col) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${r}Shape[1] + col + ${d};
        vec2 uv = uvFromFlat(${r}TexShape[0], ${r}TexShape[1], index);
        return sampleTexture(${r}, uv);
      }
    `:`
  float ${i}(int row, int col) {
    // Explicitly use integer operations as dot() only works on floats.
    int index = row * ${n[1]} + col + ${d};
    vec2 uv = uvFromFlat(${l}, ${u}, index);
    return sampleTexture(${r}, uv);
  }
`}function yw(e,t){let n=e.shapeInfo.logicalShape,r=e.name,i=`get`+r.charAt(0).toUpperCase()+r.slice(1),a=e.shapeInfo.texShape,o=[Math.ceil(a[0]/2),Math.ceil(a[1]/2)];if(n[0]===1)return`
        ${UC(kw(e,n.slice(1)),t)}
        vec4 ${i}(int b, int row, int col) {
          return ${i}(${Aw([`b`,`row`,`col`],[1,2])});
        }
      `;let s=MC();if(t)return`
    vec4 ${i}(int b, int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${r}TexShape[0]) / 2.0), ceil(float(${r}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${r}Shape[2]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${r}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom3D(
        packedTexShape[0], packedTexShape[1], texelsInBatch, valuesPerRow, b, row, col);
      return ${s.texture2D}(${r}, uv);
    }
  `;let c=o[0],l=o[1],u=Math.ceil(n[2]/2);return`
    vec4 ${i}(int b, int row, int col) {
      vec2 uv = packedUVfrom3D(
        ${c}, ${l}, ${u*Math.ceil(n[1]/2)}, ${u}, b, row, col);
      return ${s.texture2D}(${r}, uv);
    }
  `}function bw(e,t){let n=e.shapeInfo.logicalShape,r=e.name,i=`get`+r.charAt(0).toUpperCase()+r.slice(1),a=n[1]*n[2],o=n[2],{newShape:s,keptDims:c}=re(n),l=s;if(l.length<n.length)return`
        ${HC(kw(e,l),t)}
        float ${i}(int row, int col, int depth) {
          return ${i}(${Aw([`row`,`col`,`depth`],c)});
        }
      `;if(e.shapeInfo.isUniform)return`
      float ${i}(int row, int col, int depth) {
        int index = round(dot(vec3(row, col, depth),
                          vec3(${a}, ${o}, 1)));
        ${Tw(e)}
      }
    `;let u=e.shapeInfo.texShape,d=u[0],f=u[1],p=e.shapeInfo.flatOffset;if(f===a&&p==null)return t?`
      float ${i}(int row, int col, int depth) {
        int stride1 = ${r}Shape[2];
        float texR = float(row);
        float texC = dot(vec2(col, depth), vec2(stride1, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${r}TexShape[1], ${r}TexShape[0]);
        return sampleTexture(${r}, uv);
      }
    `:`
        float ${i}(int row, int col, int depth) {
          float texR = float(row);
          float texC = dot(vec2(col, depth), vec2(${o}, 1));
          vec2 uv = (vec2(texC, texR) + halfCR) /
                     vec2(${f}.0, ${d}.0);
          return sampleTexture(${r}, uv);
        }
      `;if(f===o&&p==null)return t?`
      float ${i}(int row, int col, int depth) {
        float texR = dot(vec2(row, col), vec2(${r}Shape[1], 1));
        float texC = float(depth);
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${r}TexShape[1], ${r}TexShape[0]);
        return sampleTexture(${r}, uv);
      }
    `:`
    float ${i}(int row, int col, int depth) {
      float texR = dot(vec2(row, col), vec2(${n[1]}, 1));
      float texC = float(depth);
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${f}.0, ${d}.0);
      return sampleTexture(${r}, uv);
    }
  `;let m=fw(r);return t?`
    float ${i}(int row, int col, int depth) {
      // Explicitly use integer operations as dot() only works on floats.
      int stride0 = ${r}Shape[1] * ${r}Shape[2];
      int stride1 = ${r}Shape[2];
      int index = row * stride0 + col * stride1 + depth + ${m};
      vec2 uv = uvFromFlat(${r}TexShape[0], ${r}TexShape[1], index);
      return sampleTexture(${r}, uv);
    }
    `:`
      float ${i}(int row, int col, int depth) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${a} + col * ${o} + depth + ${m};
        vec2 uv = uvFromFlat(${d}, ${f}, index);
        return sampleTexture(${r}, uv);
      }
  `}function xw(e,t){let n=e.name,r=`get`+n.charAt(0).toUpperCase()+n.slice(1),i=MC();if(t)return`
    vec4 ${r}(int b2, int b, int row, int col) {
      int valuesPerRow = int(ceil(float(${n}Shape[3]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${n}Shape[2]) / 2.0));
      int index = b * texelsInBatch + (row / 2) * valuesPerRow + (col / 2);
      texelsInBatch *= ${n}Shape[1];
      index = b2 * texelsInBatch + index;
      ivec2 packedTexShape = ivec2(ceil(float(${n}TexShape[0]) / 2.0), ceil(float(${n}TexShape[1]) / 2.0));
      int texR = index / packedTexShape[1];
      int texC = index - texR * packedTexShape[1];
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(packedTexShape[1], packedTexShape[0]); return ${i.texture2D}(${n}, uv);
    }
  `;let a=e.shapeInfo.logicalShape,o=a.length,s=e.shapeInfo.texShape,c=[Math.ceil(s[0]/2),Math.ceil(s[1]/2)],l=c[0],u=c[1],d=Math.ceil(a[o-1]/2),f=d*Math.ceil(a[o-2]/2),p=`int b, int row, int col`,m=`b * ${f} + (row / 2) * ${d} + (col / 2)`;for(let e=2;e<o-1;e++)p=`int b${e}, `+p,f*=a[o-e-1],m=`b${e} * ${f} + `+m;return`
    vec4 ${r}(${p}) {
      int index = ${m};
      int texR = index / ${u};
      int texC = index - texR * ${u};
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${u}, ${l});
      return ${i.texture2D}(${n}, uv);
    }
  `}function Sw(e,t){let n=e.shapeInfo.logicalShape,r=e.name,i=`get`+r.charAt(0).toUpperCase()+r.slice(1),a=n[3],o=n[2]*a,s=n[1]*o,{newShape:c,keptDims:l}=re(n);if(c.length<n.length)return`
      ${HC(kw(e,c),t)}
      float ${i}(int row, int col, int depth, int depth2) {
        return ${i}(${Aw([`row`,`col`,`depth`,`depth2`],l)});
      }
    `;if(e.shapeInfo.isUniform)return`
      float ${i}(int row, int col, int depth, int depth2) {
        int index = round(dot(vec4(row, col, depth, depth2),
                          vec4(${s}, ${o}, ${a}, 1)));
        ${Tw(e)}
      }
    `;let u=e.shapeInfo.flatOffset,d=e.shapeInfo.texShape,f=d[0],p=d[1],m=`int stride2 = ${r}Shape[3];`,h=`int stride1 = ${r}Shape[2] * stride2;`,g=`int stride0 = ${r}Shape[1] * stride1;`;if(p===s&&u==null)return t?`
      float ${i}(int row, int col, int depth, int depth2) {
        ${m}
        ${h}
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(stride1, stride2, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${r}TexShape[1], ${r}TexShape[0]);
        return sampleTexture(${r}, uv);
      }
    `:`
      float ${i}(int row, int col, int depth, int depth2) {
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(${o}, ${a}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${p}.0, ${f}.0);
        return sampleTexture(${r}, uv);
      }
    `;if(p===a&&u==null)return t?`
      float ${i}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${r}Shape[1] * ${r}Shape[2], ${r}Shape[2], 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${r}TexShape[1], ${r}TexShape[0]);
        return sampleTexture(${r}, uv);
      }
    `:`
      float ${i}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${n[1]*n[2]}, ${n[2]}, 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${p}.0, ${f}.0);
        return sampleTexture(${r}, uv);
      }
    `;let _=fw(r);return t?`
    float ${i}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      ${m}
      ${h}
      ${g}
      int index = row * stride0 + col * stride1 +
          depth * stride2 + depth2;
      vec2 uv = uvFromFlat(${r}TexShape[0], ${r}TexShape[1], index + ${_});
      return sampleTexture(${r}, uv);
    }
  `:`
    float ${i}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${s} + col * ${o} +
          depth * ${a} + depth2;
      vec2 uv = uvFromFlat(${f}, ${p}, index + ${_});
      return sampleTexture(${r}, uv);
    }
  `}function Cw(e){let t=e.shapeInfo.logicalShape,n=e.name,r=`get`+n.charAt(0).toUpperCase()+n.slice(1),i=t[4],a=t[3]*i,o=t[2]*a,s=t[1]*o,{newShape:c,keptDims:l}=re(t);if(c.length<t.length)return`
      ${HC(kw(e,c))}
      float ${r}(int row, int col, int depth, int depth2, int depth3) {
        return ${r}(${Aw([`row`,`col`,`depth`,`depth2`,`depth3`],l)});
      }
    `;if(e.shapeInfo.isUniform)return`
      float ${r}(int row, int col, int depth, int depth2, int depth3) {
        float index = dot(
          vec4(row, col, depth, depth2),
          vec4(${s}, ${o}, ${a}, ${i})) +
          depth3;
        ${Tw(e)}
      }
    `;let u=e.shapeInfo.flatOffset,d=e.shapeInfo.texShape,f=d[0],p=d[1];return p===s&&u==null?`
      float ${r}(int row, int col, int depth, int depth2, int depth3) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
                         vec4(${o}, ${a}, ${i}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${p}.0, ${f}.0);
        return sampleTexture(${n}, uv);
      }
    `:p===i&&u==null?`
      float ${r}(int row, int col, int depth, int depth2, int depth3) {
        float texR = dot(
          vec4(row, col, depth, depth2),
          vec4(${t[1]*t[2]*t[3]},
               ${t[2]*t[3]}, ${t[3]}, 1));
        int texC = depth3;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${p}.0, ${f}.0);
        return sampleTexture(${n}, uv);
      }
    `:`
    float ${r}(int row, int col, int depth, int depth2, int depth3) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${s} + col * ${o} + depth * ${a} +
          depth2 * ${i} + depth3 + ${fw(n)};
      vec2 uv = uvFromFlat(${f}, ${p}, index);
      return sampleTexture(${n}, uv);
    }
  `}function ww(e){let t=e.shapeInfo.logicalShape,n=e.name,r=`get`+n.charAt(0).toUpperCase()+n.slice(1),{newShape:i,keptDims:a}=re(t);if(i.length<t.length)return`
      ${HC(kw(e,i))}
      float ${r}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        return ${r}(${Aw([`row`,`col`,`depth`,`depth2`,`depth3`,`depth4`],a)});
      }
    `;let o=t[5],s=t[4]*o,c=t[3]*s,l=t[2]*c,u=t[1]*l;if(e.shapeInfo.isUniform)return`
      float ${r}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
        int index = round(dot(
          vec4(row, col, depth, depth2),
          vec4(${u}, ${l}, ${c}, ${s})) +
          dot(
            vec2(depth3, depth4),
            vec2(${o}, 1)));
        ${Tw(e)}
      }
    `;let d=e.shapeInfo.flatOffset,f=e.shapeInfo.texShape,p=f[0],m=f[1];return m===u&&d==null?`
      float ${r}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
          vec4(${l}, ${c}, ${s}, ${o})) +
               float(depth4);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${m}.0, ${p}.0);
        return sampleTexture(${n}, uv);
      }
    `:m===o&&d==null?`
      float ${r}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        float texR = dot(vec4(row, col, depth, depth2),
          vec4(${t[1]*t[2]*t[3]*t[4]},
               ${t[2]*t[3]*t[4]},
               ${t[3]*t[4]},
               ${t[4]})) + float(depth3);
        int texC = depth4;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${m}.0, ${p}.0);
        return sampleTexture(${n}, uv);
      }
    `:`
    float ${r}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${u} + col * ${l} + depth * ${c} +
          depth2 * ${s} + depth3 * ${o} + depth4 + ${fw(n)};
      vec2 uv = uvFromFlat(${p}, ${m}, index);
      return sampleTexture(${n}, uv);
    }
  `}function Tw(e){let t=e.name,n=S(e.shapeInfo.logicalShape);return n<2?`return ${t};`:`
    for (int i = 0; i < ${n}; i++) {
      if (i == index) {
        return ${t}[i];
      }
    }
  `}function Ew(e,t){let n=e.name,r=n.charAt(0).toUpperCase()+n.slice(1),i=`get`+r+`AtOutCoords`,a=e.shapeInfo.logicalShape.length,o=t.logicalShape.length,s=BC(e.shapeInfo.logicalShape,t.logicalShape),c=Z(o),l=o-a,u,d=[`x`,`y`,`z`,`w`,`u`,`v`];u=a===0?``:o<2&&s.length>=1?`coords = 0;`:s.map(e=>`coords.${d[e+l]} = 0;`).join(`
`);let f=``;f=o<2&&a>0?`coords`:e.shapeInfo.logicalShape.map((e,t)=>`coords.${d[t+l]}`).join(`, `);let p=`return outputValue;`,m=S(e.shapeInfo.logicalShape)===1,h=S(t.logicalShape)===1;if(a===1&&!m&&!h)p=`
      return vec4(outputValue.xy, outputValue.xy);
    `;else if(m&&!h)p=o===1?`
        return vec4(outputValue.x, outputValue.x, 0., 0.);
      `:`
        return vec4(outputValue.x);
      `;else if(s.length){let e=a-2,t=a-1;s.indexOf(e)>-1&&s.indexOf(t)>-1?p=`return vec4(outputValue.x);`:s.indexOf(e)>-1?p=`return vec4(outputValue.x, outputValue.y, outputValue.x, outputValue.y);`:s.indexOf(t)>-1&&(p=`return vec4(outputValue.xx, outputValue.zz);`)}return`
    vec4 ${i}() {
      ${c} coords = getOutputCoords();
      ${u}
      vec4 outputValue = get${r}(${f});
      ${p}
    }
  `}function Dw(e,t){let n=e.name,r=n.charAt(0).toUpperCase()+n.slice(1),i=`get`+r+`AtOutCoords`,a=t.texShape,o=e.shapeInfo.texShape,s=e.shapeInfo.logicalShape.length,c=t.logicalShape.length;if(!e.shapeInfo.isUniform&&s===c&&e.shapeInfo.flatOffset==null&&w(o,a))return`
      float ${i}() {
        return sampleTexture(${n}, resultUV);
      }
    `;let l=Z(c),u=BC(e.shapeInfo.logicalShape,t.logicalShape),d=c-s,f,p=[`x`,`y`,`z`,`w`,`u`,`v`];f=s===0?``:c<2&&u.length>=1?`coords = 0;`:u.map(e=>`coords.${p[e+d]} = 0;`).join(`
`);let m=``;return m=c<2&&s>0?`coords`:e.shapeInfo.logicalShape.map((e,t)=>`coords.${p[t+d]}`).join(`, `),`
    float ${i}() {
      ${l} coords = getOutputCoords();
      ${f}
      return get${r}(${m});
    }
  `}function Z(e){if(e<=1)return`int`;if(e===2)return`ivec2`;if(e===3)return`ivec3`;if(e===4)return`ivec4`;if(e===5)return`ivec5`;if(e===6)return`ivec6`;throw Error(`GPU for rank ${e} is not yet supported`)}function Ow(e,t,n){let{newShape:r,keptDims:i}=re(t),a=t.length,o=e&&a===3&&t[0]===1,s=o?t.slice(1):r,c=!e&&a>1&&!w(t,n)&&r.length<a||o;return{useSqueezeShape:c,uniformShape:c?s:t,keptDims:i}}function kw(e,t){let n=JSON.parse(JSON.stringify(e));return n.shapeInfo.logicalShape=t,n}function Aw(e,t){return t.map(t=>e[t]).join(`, `)}function jw(e,t,n,r){let i=n.map((e,n)=>{let r={logicalShape:e.shape,texShape:e.isUniform?null:e.texData.texShape,isUniform:e.isUniform,isPacked:e.isUniform?!1:e.texData.isPacked,flatOffset:null};return e.texData!=null&&e.texData.slice!=null&&e.texData.slice.flatOffset>0&&(r.flatOffset=e.texData.slice.flatOffset),{name:t.variableNames[n],shapeInfo:r}}),a=i.map(e=>e.shapeInfo),o={logicalShape:r.shape,texShape:r.texData.texShape,isUniform:!1,isPacked:r.texData.isPacked,flatOffset:null},s=VC(i,o,t),c=GS(e.gl,s),l=e.createProgram(c);return O().get(`ENGINE_COMPILE_ONLY`)?{program:t,fragmentShader:c,source:s,webGLProgram:l,inShapeInfos:a,outShapeInfo:o,variablesLocations:null,customUniformLocations:null,infLoc:null,nanLoc:null,outShapeLocation:null,outShapeStridesLocation:null,outTexShapeLocation:null}:(e.buildVao(l),Object.assign({program:t,fragmentShader:c,source:s,webGLProgram:l,inShapeInfos:a,outShapeInfo:o},Mw(e,t,l)))}function Mw(e,t,n){let r=[],i=[],a,o,s,c=null,l=null;l=e.getUniformLocation(n,`NAN`,!1),O().getNumber(`WEBGL_VERSION`)===1&&(c=e.getUniformLocation(n,`INFINITY`,!1));for(let i of t.variableNames){let a={name:i,uniform:e.getUniformLocation(n,i,!1),offset:e.getUniformLocation(n,`offset${i}`,!1)};t.enableShapeUniforms&&(a.shape=e.getUniformLocation(n,`${i}Shape`,!1),a.texShape=e.getUniformLocation(n,`${i}TexShape`,!1)),r.push(a)}if(t.enableShapeUniforms&&(a=e.getUniformLocation(n,`outShape`,!1),s=e.getUniformLocation(n,`outShapeStrides`,!1),o=e.getUniformLocation(n,`outTexShape`,!1)),t.customUniforms)for(let r of t.customUniforms)i.push(e.getUniformLocation(n,r.name,!1));return{variablesLocations:r,customUniformLocations:i,infLoc:c,nanLoc:l,outShapeLocation:a,outShapeStridesLocation:s,outTexShapeLocation:o}}function Nw(e,t){if(e.length!==t.length)throw Error(`Binary was compiled with ${e.length} inputs, but was executed with ${t.length} inputs`);e.forEach((e,n)=>{let r=e.logicalShape,i=t[n],a=i.shape;if(!w(r,a))throw Error(`Binary was compiled with different shapes than the current args. Shapes ${r} and ${a} must match`);if(e.isUniform&&i.isUniform)return;let o=e.texShape,s=i.isUniform?null:i.texData.texShape;if(!w(o,s))throw Error(`Binary was compiled with different texture shapes than the current args. Shape ${o} and ${s} must match`)})}function Pw(e,t,n,r,i){t.program.enableShapeUniforms||(Nw(t.inShapeInfos,n),Nw([t.outShapeInfo],[r]));let a=r.texData.texture,o=r.texData.texShape;r.texData.isPacked?e.setOutputPackedMatrixTexture(a.texture,o[0],o[1]):e.setOutputMatrixTexture(a.texture,o[0],o[1]),e.setProgram(t.webGLProgram),e.bindVertexArray(t.webGLProgram.vao),O().getNumber(`WEBGL_VERSION`)===1&&t.infLoc!==null&&e.gl.uniform1f(t.infLoc,1/0),t.nanLoc!==null&&e.gl.uniform1f(t.nanLoc,NaN);for(let r=0;r<n.length;++r){let i=n[r],{uniform:a,offset:o,shape:s,texShape:c}=t.variablesLocations[r];if(s){let{uniformShape:n}=Ow(t.program.packedInputs,i.shape,i.texData.texShape);switch(n.length){case 1:e.gl.uniform1iv(s,new Int32Array(n));break;case 2:e.gl.uniform2iv(s,new Int32Array(n));break;case 3:e.gl.uniform3iv(s,new Int32Array(n));break;case 4:e.gl.uniform4iv(s,new Int32Array(n));break;default:break}}if(c&&e.gl.uniform2i(c,i.texData.texShape[0],i.texData.texShape[1]),a!=null){if(i.isUniform){if(S(i.shape)<2)e.gl.uniform1f(a,i.uniformValues[0]);else{let t=i.uniformValues;t instanceof Float32Array||(t=new Float32Array(t)),e.gl.uniform1fv(a,t)}continue}i.texData.slice!=null&&o!=null&&e.gl.uniform1i(o,i.texData.slice.flatOffset),e.setInputMatrixTexture(i.texData.texture.texture,a,r)}}let s=t.outShapeLocation;if(s)switch(r.shape.length){case 1:e.gl.uniform1iv(s,new Int32Array(r.shape));break;case 2:e.gl.uniform2iv(s,new Int32Array(r.shape));break;case 3:e.gl.uniform3iv(s,new Int32Array(r.shape));break;case 4:e.gl.uniform4iv(s,new Int32Array(r.shape));break;default:break}if(t.outShapeStridesLocation){let n=_e(r.shape);switch(r.shape.length){case 2:e.gl.uniform1iv(t.outShapeStridesLocation,new Int32Array(n));break;case 3:e.gl.uniform2iv(t.outShapeStridesLocation,new Int32Array(n));break;case 4:e.gl.uniform3iv(t.outShapeStridesLocation,new Int32Array(n));break;default:break}}if(t.outTexShapeLocation&&e.gl.uniform2i(t.outTexShapeLocation,r.texData.texShape[0],r.texData.texShape[1]),t.program.customUniforms&&i)for(let n=0;n<t.program.customUniforms.length;++n){let r=t.program.customUniforms[n],a=t.customUniformLocations[n],o=i[n];if(r.type===`float`)e.gl.uniform1fv(a,o);else if(r.type===`vec2`)e.gl.uniform2fv(a,o);else if(r.type===`vec3`)e.gl.uniform3fv(a,o);else if(r.type===`vec4`)e.gl.uniform4fv(a,o);else if(r.type===`int`)e.gl.uniform1iv(a,o);else if(r.type===`ivec2`)e.gl.uniform2iv(a,o);else if(r.type===`ivec3`)e.gl.uniform3iv(a,o);else if(r.type===`ivec4`)e.gl.uniform4iv(a,o);else throw Error(`uniform type ${r.type} is not supported yet.`)}e.executeProgram()}function Fw(e,t,n){let r=``;t.concat(n).forEach(t=>{let i=t.texData!=null&&t.texData.slice!=null&&t.texData.slice.flatOffset>0;if(e.enableShapeUniforms&&!t.isUniform){let a=t.texData.texShape,{useSqueezeShape:o,uniformShape:s,keptDims:c}=Ow(e.packedInputs,t.shape,a),l=``,u=``,d=``;if(s.length===1&&e.packedInputs){let e=[Math.ceil(a[0]/2),Math.ceil(a[1]/2)];l=`${e[0]>1}_${e[1]>1}`}else if(s.length===2&&!e.packedInputs)u=`${s[0]>1}_${s[1]>1}`;else if(s.length>2&&!e.packedInputs){let e=_e(s);d=`${e[0]===a[1]}_${e[e.length-1]===a[1]}`}let f=t.shape.length,p=s.length===2&&w(t.shape,a),m=S(t.shape)===1,h=rl(t.shape,n.shape),g=!e.packedInputs&&f===n.shape.length&&w(a,n.texData.texShape),_=e.packedInputs||s.length>2?``:`${a[0]>1}_${a[1]>1}`;r+=`${f}_${g}_${o?c:``}_${s.length}_${m}_${h}_${p}_${l}_${u}_${d}_${_}_${i}`}else{let e=t.isUniform?`uniform`:t.texData.texShape;r+=`${t.shape}_${e}_${i}`}});let i=e.userCode,a=e.constructor.name;return a+=`_`+r+`_`+i+`${O().getNumber(`WEBGL_VERSION`)}`,a}function Iw(e){return O().getBool(`WEBGL_USE_SHAPES_UNIFORMS`)&&e<=4}var Lw=class{constructor(e){this.variableNames=[`A`],this.packedInputs=!1,this.packedOutput=!0,this.outPackingScheme=kS.DENSE,this.customUniforms=[{name:`texShape`,type:`ivec2`}];let t=MC();this.outputShape=e,this.enableShapeUniforms=Iw(this.outputShape.length),this.userCode=`
      ivec3 outCoordsFromFlatIndex(int index) {
        ${this.enableShapeUniforms?PC([`r`,`c`,`d`],e):NC([`r`,`c`,`d`],e)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getA(rc.x, rc.y, rc.z);
        }

        ${t.output} = result;
      }
    `}},Rw=class{constructor(e){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.outPackingScheme=kS.DENSE,this.customUniforms=[{name:`texShape`,type:`ivec2`}];let t=MC();this.outputShape=e,this.enableShapeUniforms=Iw(this.outputShape.length),this.userCode=`
      ivec3 outCoordsFromFlatIndex(int index) {
        ${this.enableShapeUniforms?PC([`r`,`c`,`d`],e):NC([`r`,`c`,`d`],e)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getChannel(getA(rc.x, rc.y, rc.z), vec2(rc.y, rc.z));
        }

        ${t.output} = result;
      }
    `}},zw=class{constructor(e){this.variableNames=[`A`],this.outTexUsage=AS.DOWNLOAD;let t=MC();this.outputShape=e,this.userCode=`
      ${zC}

      void main() {
        float x = getAAtOutCoords();
        ${t.output} = encode_float(x);
      }
    `}},Bw=class{constructor(e){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!1,this.outTexUsage=AS.DOWNLOAD;let t=MC();this.outputShape=e,this.userCode=`
      ${zC}

      void main() {
        ivec3 coords = getOutputCoords();
        float x = getChannel(getAAtOutCoords(), vec2(coords.y, coords.z));
        ${t.output} = encode_float(x);
      }
    `}},Vw={R:0,G:1,B:2,A:3},Hw=class{constructor(e,t=!1,n=`RGBA`){this.variableNames=[`A`],this.customUniforms=[{name:`texShape`,type:`ivec2`}];let r=MC();this.outputShape=e,this.enableShapeUniforms=Iw(this.outputShape.length);let i=`result`;t&&(i=`floor(result * 255. + 0.5)`);let a=``;for(let e=0;e<n.length;e++){let t=n[e];a+=`
          if(offset == ${e}) {
            result = values[${Vw[t]}];
          }`}this.userCode=`
      ${this.enableShapeUniforms?RC():LC(e)}

      void main() {
        ivec3 coords = getOutputCoords();
        int flatIndex = getFlatIndex(coords);
        float result = 0.;
        int offset = imod(flatIndex, ${n.length});

        flatIndex = idiv(flatIndex, ${n.length}, 1.);

        int r = flatIndex / texShape[1];
        if (r < texShape[0]) {
          int c = imod(flatIndex, texShape[1]);
          vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
          vec4 values = ${r.texture2D}(A, uv);
          ${a}
        }
        ${r.output} = vec4(${i}, 0., 0., 0.);
      }
    `}},Uw=class{constructor(e,t=!1){this.variableNames=[`A`],this.packedInputs=!1,this.packedOutput=!0,this.customUniforms=[{name:`texShape`,type:`ivec2`}];let n=MC();this.outputShape=e,this.enableShapeUniforms=Iw(this.outputShape.length);let r=``,i=`result`;t&&(i=`floor(result * 255. + 0.5)`);for(let t=0;t<=1;t++)for(let i=0;i<=1;i++){let a=t*2+i;r+=`
          localCoords = coords;
          if(localCoords[2] + ${i} < ${this.enableShapeUniforms?`outShape[2]`:`${e[2]}`}) {
          localCoords[2] += ${i};
          if (localCoords[1] + ${t} < ${this.enableShapeUniforms?`outShape[1]`:`${e[1]}`}) {
            localCoords[1] += ${t};

            flatIndex = getFlatIndex(localCoords);
            offset = imod(flatIndex, 4);

            flatIndex = idiv(flatIndex, 4, 1.);

            int r = flatIndex / texShape[1];
            int c = imod(flatIndex, texShape[1]);
            vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
            values = ${n.texture2D}(A, uv);

            if (offset == 0) {
              result[${a}] = values[0];
            } else if (offset == 1) {
              result[${a}] = values[1];
            } else if (offset == 2) {
              result[${a}] = values[2];
            } else {
              result[${a}] = values[3];
            }
          }
        }
        `}this.userCode=`
        ${this.enableShapeUniforms?RC():LC(e)}

        void main() {
          ivec3 coords = getOutputCoords();

          vec4 result = vec4(0.);
          int flatIndex, r, c, offset;
          ivec3 localCoords;
          vec2 uv;
          vec4 values;

          ${r}

          ${n.output} = ${i};
        }
    `}};function Ww(e){let t=MC();return WS(e,`${t.version}
    precision highp float;
    ${t.attribute} vec3 clipSpacePos;
    ${t.attribute} vec2 uv;
    ${t.varyingVs} vec2 resultUV;

    void main() {
      gl_Position = vec4(clipSpacePos, 1);
      resultUV = uv;
    }`)}function Gw(e){return ZS(e,new Float32Array([-1,1,0,0,1,-1,-1,0,0,0,1,1,0,1,1,1,-1,0,1,0]))}function Kw(e){return QS(e,new Uint16Array([0,1,2,2,1,3]))}function qw(e,t,n,r,i,a){eC(t,n);let o=$S(e),s=e.TEXTURE_2D;return Y(e,()=>e.bindTexture(s,o)),Y(e,()=>e.texParameteri(s,e.TEXTURE_WRAP_S,e.CLAMP_TO_EDGE)),Y(e,()=>e.texParameteri(s,e.TEXTURE_WRAP_T,e.CLAMP_TO_EDGE)),Y(e,()=>e.texParameteri(s,e.TEXTURE_MIN_FILTER,e.NEAREST)),Y(e,()=>e.texParameteri(s,e.TEXTURE_MAG_FILTER,e.NEAREST)),O().getNumber(`WEBGL_VERSION`)===1?Y(e,()=>e.texImage2D(s,0,r,t,n,0,i,a,null)):Y(e,()=>e.texStorage2D(s,1,r,t,n)),Y(e,()=>e.bindTexture(e.TEXTURE_2D,null)),{texture:o,texShape:[n,t]}}function Jw(e){return e.internalFormatFloat}function Yw(e,t,n,r){let[i,a]=MS(t,n);return qw(e,i,a,Jw(r),r.textureFormatFloat,e.FLOAT)}function Xw(e){return e.internalFormatHalfFloat}function Zw(e,t,n,r){let[i,a]=MS(t,n);return qw(e,i,a,Xw(r),r.textureFormatFloat,r.textureTypeHalfFloat)}function Qw(e){return e.downloadTextureFormat}function $w(e,t,n,r){let[i,a]=MS(t,n);return qw(e,i,a,Qw(r),e.RGBA,e.UNSIGNED_BYTE)}function eT(e){return e.internalFormatPackedFloat}function tT(e,t,n,r){let[i,a]=FS(t,n);return qw(e,i,a,eT(r),e.RGBA,e.FLOAT)}function nT(e){return e.internalFormatPackedHalfFloat}function rT(e,t,n,r){let[i,a]=FS(t,n);return qw(e,i,a,nT(r),e.RGBA,r.textureTypeHalfFloat)}function iT(e,t,n){return Y(e,()=>e.bindBuffer(e.ARRAY_BUFFER,n)),nC(e,t,`clipSpacePos`,n,3,20,0)&&nC(e,t,`uv`,n,2,20,12)}function aT(e,t,n,r,i,a){Y(e,()=>e.bindTexture(e.TEXTURE_2D,t));let o,s,c;i instanceof Uint8Array?(o=new Uint8Array(n*r*4),s=e.UNSIGNED_BYTE,c=e.RGBA):(o=new Float32Array(n*r*4),s=e.FLOAT,c=a.internalFormatPackedFloat),o.set(i),O().getNumber(`WEBGL_VERSION`)===2?Y(e,()=>e.texSubImage2D(e.TEXTURE_2D,0,0,0,n,r,e.RGBA,s,o)):Y(e,()=>e.texImage2D(e.TEXTURE_2D,0,c,n,r,0,e.RGBA,s,o)),Y(e,()=>e.bindTexture(e.TEXTURE_2D,null))}function oT(e,t,n){Y(e,()=>e.bindTexture(e.TEXTURE_2D,t)),n.data instanceof Uint8Array?O().getNumber(`WEBGL_VERSION`)===2?Y(e,()=>e.texSubImage2D(e.TEXTURE_2D,0,0,0,n.width,n.height,e.RGBA,e.UNSIGNED_BYTE,n.data)):Y(e,()=>e.texImage2D(e.TEXTURE_2D,0,e.RGBA,n.width,n.height,0,e.RGBA,e.UNSIGNED_BYTE,n.data)):O().getNumber(`WEBGL_VERSION`)===2?Y(e,()=>e.texSubImage2D(e.TEXTURE_2D,0,0,0,e.RGBA,e.UNSIGNED_BYTE,n)):Y(e,()=>e.texImage2D(e.TEXTURE_2D,0,e.RGBA,e.RGBA,e.UNSIGNED_BYTE,n)),Y(e,()=>e.bindTexture(e.TEXTURE_2D,null))}function sT(e,t,n,r){let i=e.createBuffer();Y(e,()=>e.bindBuffer(e.PIXEL_PACK_BUFFER,i));let a=16*t*n;return Y(e,()=>e.bufferData(e.PIXEL_PACK_BUFFER,a,e.STREAM_READ)),Y(e,()=>e.readPixels(0,0,n,t,e.RGBA,e.FLOAT,0)),Y(e,()=>e.bindBuffer(e.PIXEL_PACK_BUFFER,null)),i}function cT(e,t,n){let r=e,i=new Float32Array(n);return r.bindBuffer(r.PIXEL_PACK_BUFFER,t),r.getBufferSubData(r.PIXEL_PACK_BUFFER,0,i),r.bindBuffer(r.PIXEL_PACK_BUFFER,null),i}function lT(e,t,n,r){let[i,a]=MS(t,n),o=new Uint8Array(NS(t*n,4));return Y(e,()=>e.readPixels(0,0,i,a,r.downloadTextureFormat,e.UNSIGNED_BYTE,o)),new Float32Array(o.buffer)}function uT(e,t,n,r,i,a,o,s){let c=e,l=new Float32Array(IS(a,o));return c.bindBuffer(c.PIXEL_PACK_BUFFER,t),c.getBufferSubData(c.PIXEL_PACK_BUFFER,0,l),c.bindBuffer(c.PIXEL_PACK_BUFFER,null),l}function dT(e,t,n){let r=new Float32Array(t*n*4);return Y(e,()=>e.readPixels(0,0,n,t,e.RGBA,e.FLOAT,r)),r}var fT=class{constructor(e){this.outputTexture=null,this.program=null,this.disposed=!1,this.itemsToPoll=[];let t=O().getNumber(`WEBGL_VERSION`);if(e==null?this.gl=ES(t):(this.gl=e,TS(t,e)),e=this.gl,O().getNumber(`WEBGL_VERSION`)===2){let t=e;this.createVertexArray=()=>Y(t,()=>t.createVertexArray()),this.bindVertexArray=e=>Y(t,()=>t.bindVertexArray(e)),this.deleteVertexArray=e=>Y(t,()=>t.deleteVertexArray(e)),this.getVertexArray=()=>Y(t,()=>t.getParameter(t.VERTEX_ARRAY_BINDING))}else if(e!=null){let t=e.getExtension(`OES_vertex_array_object`);if(t==null)throw Error(`All WebGL1 implementations are expected to offer OES_vertex_array_object.`);this.createVertexArray=()=>Y(e,()=>t.createVertexArrayOES()),this.bindVertexArray=n=>Y(e,()=>t.bindVertexArrayOES(n)),this.deleteVertexArray=n=>Y(e,()=>t.deleteVertexArrayOES(n)),this.getVertexArray=()=>Y(e,()=>e.getParameter(t.VERTEX_ARRAY_BINDING_OES))}let n=`WEBGL_color_buffer_float`,r=`EXT_color_buffer_half_float`;if(this.parallelCompilationExtension=this.gl.getExtension(`KHR_parallel_shader_compile`),O().getNumber(`WEBGL_VERSION`)===1){let e=`OES_texture_half_float`;if(this.textureFloatExtension=US(this.gl,`OES_texture_float`),wC(this.gl,e))this.textureHalfFloatExtension=US(this.gl,e);else if(O().get(`WEBGL_FORCE_F16_TEXTURES`))throw Error(`GL context does not support half float textures, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.`);if(this.colorBufferFloatExtension=this.gl.getExtension(n),wC(this.gl,r))this.colorBufferHalfFloatExtension=US(this.gl,r);else if(O().get(`WEBGL_FORCE_F16_TEXTURES`))throw Error(`GL context does not support color renderable half floats, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.`)}else if(n=`EXT_color_buffer_float`,wC(this.gl,n))this.colorBufferFloatExtension=this.gl.getExtension(n);else if(wC(this.gl,r))this.colorBufferHalfFloatExtension=this.gl.getExtension(r);else throw Error(`GL context does not support color renderable floats`);this.vertexBuffer=Gw(this.gl),this.indexBuffer=Kw(this.gl),this.framebuffer=tC(this.gl),this.textureConfig=LS(this.gl,this.textureHalfFloatExtension)}get debug(){return O().getBool(`DEBUG`)}dispose(){if(this.disposed)return;this.program!=null&&console.warn(`Disposing a GPGPUContext that still has a bound WebGLProgram. This is probably a resource leak, delete the program with GPGPUContext.deleteProgram before disposing.`),this.outputTexture!=null&&console.warn(`Disposing a GPGPUContext that still has a bound output matrix texture.  This is probably a resource leak, delete the output matrix texture with GPGPUContext.deleteMatrixTexture before disposing.`);let e=this.gl;Y(e,()=>e.finish()),Y(e,()=>e.bindFramebuffer(e.FRAMEBUFFER,null)),Y(e,()=>e.deleteFramebuffer(this.framebuffer)),Y(e,()=>e.bindBuffer(e.ARRAY_BUFFER,null)),Y(e,()=>e.bindBuffer(e.ELEMENT_ARRAY_BUFFER,null)),Y(e,()=>e.deleteBuffer(this.indexBuffer)),this.disposed=!0}createFloat32MatrixTexture(e,t){return this.throwIfDisposed(),Yw(this.gl,e,t,this.textureConfig)}createFloat16MatrixTexture(e,t){return this.throwIfDisposed(),Zw(this.gl,e,t,this.textureConfig)}createUnsignedBytesMatrixTexture(e,t){return this.throwIfDisposed(),$w(this.gl,e,t,this.textureConfig)}uploadPixelDataToTexture(e,t){this.throwIfDisposed(),oT(this.gl,e,t)}uploadDenseMatrixToTexture(e,t,n,r){this.throwIfDisposed(),aT(this.gl,e,t,n,r,this.textureConfig)}createFloat16PackedMatrixTexture(e,t){return this.throwIfDisposed(),rT(this.gl,e,t,this.textureConfig)}createPackedMatrixTexture(e,t){return this.throwIfDisposed(),tT(this.gl,e,t,this.textureConfig)}deleteMatrixTexture(e){this.throwIfDisposed(),this.outputTexture===e&&(cC(this.gl,this.framebuffer),this.outputTexture=null),Y(this.gl,()=>this.gl.deleteTexture(e))}downloadByteEncodedFloatMatrixFromOutputTexture(e,t,n){return this.downloadMatrixDriver(e,()=>lT(this.gl,t,n,this.textureConfig))}downloadPackedMatrixFromBuffer(e,t,n,r,i,a){return uT(this.gl,e,t,n,r,i,a,this.textureConfig)}downloadFloat32MatrixFromBuffer(e,t){return cT(this.gl,e,t)}createBufferFromTexture(e,t,n){this.bindTextureToFrameBuffer(e);let r=sT(this.gl,t,n,this.textureConfig);return this.unbindTextureToFrameBuffer(),r}createAndWaitForFence(){let e=this.createFence(this.gl);return this.pollFence(e)}createFence(e){let t,n;if(O().getBool(`WEBGL_FENCE_API_ENABLED`)){let r=e,i=r.fenceSync(r.SYNC_GPU_COMMANDS_COMPLETE,0);e.flush(),n=()=>{let e=r.clientWaitSync(i,0,0);return e===r.ALREADY_SIGNALED||e===r.CONDITION_SATISFIED},t=i}else O().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`)>0?(t=this.beginQuery(),this.endQuery(),n=()=>this.isQueryAvailable(t,O().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`))):n=()=>!0;return{query:t,isFencePassed:n}}downloadMatrixFromPackedTexture(e,t,n){return this.downloadMatrixDriver(e,()=>dT(this.gl,t,n))}createProgram(e){this.throwIfDisposed();let t=this.gl;this.vertexShader??=Ww(t);let n=JS(t);Y(t,()=>t.attachShader(n,this.vertexShader)),Y(t,()=>t.attachShader(n,e)),YS(t,n);let r=Object.assign(n,{vao:this.createVertexArray()});return this.debug&&XS(t,r),r}buildVao(e){this.setProgram(e),this.bindVertexArray(e.vao);let t=this.gl;Y(t,()=>t.bindBuffer(t.ELEMENT_ARRAY_BUFFER,this.indexBuffer)),iT(t,e,this.vertexBuffer)}deleteProgram(e){this.throwIfDisposed(),e===this.program&&(this.program=null),e!=null&&(Y(this.gl,()=>this.gl.deleteProgram(e)),this.deleteVertexArray(e.vao))}setProgram(e){this.throwIfDisposed(),this.program=e,this.program!=null&&this.debug&&XS(this.gl,this.program),Y(this.gl,()=>this.gl.useProgram(e))}getUniformLocation(e,t,n=!0){return this.throwIfDisposed(),n?iC(this.gl,e,t):aC(this.gl,e,t)}getAttributeLocation(e,t){return this.throwIfDisposed(),Y(this.gl,()=>this.gl.getAttribLocation(e,t))}getUniformLocationNoThrow(e,t){return this.throwIfDisposed(),this.gl.getUniformLocation(e,t)}setInputMatrixTexture(e,t,n){this.throwIfDisposed(),this.throwIfNoProgram(),oC(this.gl,e,t,n)}setOutputMatrixTexture(e,t,n){this.setOutputMatrixTextureDriver(e,n,t)}setOutputPackedMatrixTexture(e,t,n){this.throwIfDisposed();let[r,i]=FS(t,n);this.setOutputMatrixTextureDriver(e,r,i)}setOutputMatrixWriteRegion(e,t,n,r){this.setOutputMatrixWriteRegionDriver(n,e,r,t)}setOutputPackedMatrixWriteRegion(e,t,n,r){throw Error(`setOutputPackedMatrixWriteRegion not implemented.`)}debugValidate(){this.program!=null&&XS(this.gl,this.program),lC(this.gl)}executeProgram(){this.throwIfDisposed(),this.throwIfNoProgram();let e=this.gl;if(this.debug){let e=this.getVertexArray();console.assert(e===this.program.vao,`VAO changed between setProgram and executeProgram!`),this.debugValidate()}Y(e,()=>e.drawElements(e.TRIANGLES,6,e.UNSIGNED_SHORT,0))}blockUntilAllProgramsCompleted(){this.throwIfDisposed(),Y(this.gl,()=>this.gl.finish())}getQueryTimerExtension(){return this.disjointQueryTimerExtension??=US(this.gl,O().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`)===2?`EXT_disjoint_timer_query_webgl2`:`EXT_disjoint_timer_query`),this.disjointQueryTimerExtension}getQueryTimerExtensionWebGL2(){return this.getQueryTimerExtension()}getQueryTimerExtensionWebGL1(){return this.getQueryTimerExtension()}beginQuery(){if(O().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`)===2){let e=this.gl,t=this.getQueryTimerExtensionWebGL2(),n=e.createQuery();return e.beginQuery(t.TIME_ELAPSED_EXT,n),n}let e=this.getQueryTimerExtensionWebGL1(),t=e.createQueryEXT();return e.beginQueryEXT(e.TIME_ELAPSED_EXT,t),t}endQuery(){if(O().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`)===2){let e=this.gl,t=this.getQueryTimerExtensionWebGL2();e.endQuery(t.TIME_ELAPSED_EXT);return}let e=this.getQueryTimerExtensionWebGL1();e.endQueryEXT(e.TIME_ELAPSED_EXT)}async waitForQueryAndGetTime(e){return await te(()=>this.disposed||this.isQueryAvailable(e,O().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`))),this.getQueryTime(e,O().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`))}getQueryTime(e,t){if(t===0)return null;if(t===2){let t=this.gl;return t.getQueryParameter(e,t.QUERY_RESULT)/1e6}else{let t=this.getQueryTimerExtensionWebGL1();return t.getQueryObjectEXT(e,t.QUERY_RESULT_EXT)/1e6}}isQueryAvailable(e,t){if(t===0)return!0;if(t===2){let t=this.gl,n=this.getQueryTimerExtensionWebGL2(),r=t.getQueryParameter(e,t.QUERY_RESULT_AVAILABLE);return this.disjoint??=this.gl.getParameter(n.GPU_DISJOINT_EXT),r&&!this.disjoint}else{let t=this.getQueryTimerExtensionWebGL1(),n=t.getQueryObjectEXT(e,t.QUERY_RESULT_AVAILABLE_EXT);return this.disjoint??=this.gl.getParameter(t.GPU_DISJOINT_EXT),n&&!this.disjoint}}pollFence(e){return new Promise(t=>{this.addItemToPoll(()=>e.isFencePassed(),()=>t())})}pollItems(){let e=pT(this.itemsToPoll.map(e=>e.isDoneFn));for(let t=0;t<=e;++t){let{resolveFn:e}=this.itemsToPoll[t];e()}this.itemsToPoll=this.itemsToPoll.slice(e+1)}addItemToPoll(e,t){if(this.itemsToPoll.push({isDoneFn:e,resolveFn:t}),this.itemsToPoll.length>1)return;let n;`setTimeoutCustom`in O().platform&&(n=O().platform.setTimeoutCustom.bind(O().platform)),te(()=>(this.pollItems(),this.itemsToPoll.length===0),()=>0,null,n)}bindTextureToFrameBuffer(e){this.throwIfDisposed(),sC(this.gl,e,this.framebuffer),this.debug&&lC(this.gl)}unbindTextureToFrameBuffer(){this.outputTexture==null?cC(this.gl,this.framebuffer):(sC(this.gl,this.outputTexture,this.framebuffer),this.debug&&lC(this.gl))}downloadMatrixDriver(e,t){this.bindTextureToFrameBuffer(e);let n=t();return this.unbindTextureToFrameBuffer(),n}setOutputMatrixTextureDriver(e,t,n){this.throwIfDisposed();let r=this.gl;sC(r,e,this.framebuffer),this.debug&&lC(r),this.outputTexture=e,Y(r,()=>r.viewport(0,0,t,n)),Y(r,()=>r.scissor(0,0,t,n))}setOutputMatrixWriteRegionDriver(e,t,n,r){this.throwIfDisposed(),Y(this.gl,()=>this.gl.scissor(e,t,n,r))}throwIfDisposed(){if(this.disposed)throw Error(`Attempted to use disposed GPGPUContext.`)}throwIfNoProgram(){if(this.program==null)throw Error(`No GPU program is currently set.`)}};function pT(e){let t=0;for(;t<e.length&&e[t]();++t);return t-1}function mT(e,t){Array.isArray(e)||(e=[e]),e.forEach(e=>{e!=null&&y(e.dtype!==`complex64`,()=>`${t} does not support complex64 tensors in the CPU backend.`)})}function hT(e){let t=new Float32Array(e.length);for(let n=0;n<e.length;++n)t[n]=Math.abs(e[n]);return t}function gT(e){return(t,n,r,i,a)=>{let o=W(t,n),s=o.length,c=_e(o),l=ie(a,S(o)),u=t.length,d=n.length,f=_e(t),p=_e(n),m=rl(t,o),h=rl(n,o);if(m.length+h.length===0)for(let t=0;t<l.length;++t)l[t]=e(r[t%r.length],i[t%i.length]);else for(let t=0;t<l.length;++t){let n=we(t,s,c),a=n.slice(-u);m.forEach(e=>a[e]=0);let o=Ce(a,u,f),g=n.slice(-d);h.forEach(e=>g[e]=0);let _=Ce(g,d,p);l[t]=e(r[o],i[_])}return[l,o]}}function _T(e){let{inputs:t,backend:n}=e,{real:r,imag:i}=t,a=n.data.get(r.dataId).values,o=n.data.get(i.dataId).values,s=n.makeTensorInfo(r.shape,`complex64`),c=n.data.get(s.dataId);return c.complexTensorInfos={real:n.makeTensorInfo(r.shape,`float32`,a),imag:n.makeTensorInfo(i.shape,`float32`,o)},s}function vT(e,t,n=`float32`){if(n===`complex64`)return _T({inputs:{real:vT(e,t,`float32`),imag:vT(e,t,`float32`)},backend:e});let r=xe(S(t),n);return e.makeTensorInfo(t,n,r)}function yT(e){let{inputs:t,backend:n}=e,{x:r}=t;return n.incRef(r.dataId),{dataId:r.dataId,shape:r.shape,dtype:r.dtype}}function bT(e){let{inputs:t,backend:n}=e,{input:r}=t,i=n.data.get(r.dataId).complexTensorInfos.real,a=n.data.get(i.dataId).values;return n.makeTensorInfo(i.shape,i.dtype,a)}function xT(e,t,n,r){if(r===`int32`)return[t,`int32`,Int32Array.from(e)];if(r===`bool`){let r=ti([0],n),[i,a]=gT((e,t)=>e===t?0:1)(t,[],e,r,`bool`);return[a,`bool`,i]}throw Error(`Error in Cast: failed to cast ${n} to ${r}`)}function ST(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{dtype:a}=r;if(a===`complex64`){if(i.dtype===`complex64`)return yT({inputs:{x:i},backend:n});let e=vT(n,i.shape,i.dtype),t=ST({inputs:{x:i},backend:n,attrs:{dtype:`float32`}}),r=_T({inputs:{real:t,imag:e},backend:n});return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),r}if(i.dtype===`complex64`){let e=bT({inputs:{input:i},backend:n}),t=ST({inputs:{x:e},backend:n,attrs:{dtype:a}});return n.disposeIntermediateTensorInfo(e),t}if(!ce(i.dtype,a)){let e=yT({inputs:{x:i},backend:n});return{dataId:e.dataId,shape:e.shape,dtype:a}}let o=n.data.get(i.dataId).values,[s,c,l]=xT(o,i.shape,i.dtype,a);return n.makeTensorInfo(s,c,l)}var CT=gT(((e,t)=>e+t));function wT(e,t,n,r,i){let a=S(r),o=xe(i,n);for(let n=0;n<e.length;n++){let r=e[n];if(r<0)throw Error(`Input x must be non-negative!`);r>=i||(a>0?o[r]+=t[n]:o[r]+=1)}return o}function TT(e,t,n,r=!1){let i=e.shape[0],a=e.shape[1],o=I([i,n],t.dtype);for(let s=0;s<i;s++)for(let i=0;i<a;i++){let a=e.get(s,i);if(a<0)throw Error(`Input x must be non-negative!`);a>=n||(r?o.set(1,s,a):t.size>0?o.set(o.get(s,a)+t.get(s,i),s,a):o.set(o.get(s,a)+1,s,a))}return o}var ET=gT(((e,t)=>e&t));function DT(e){return(t,n,r)=>{let i=ae(n,t.length);for(let n=0;n<t.length;++n)i[n]=e(t[n],r);return i}}function OT(e,t,n){return kT(e,DT(t),n)}function kT(e,t,n){return({inputs:r,attrs:i,backend:a})=>{let{x:o}=r;mT(o,e);let s=a,c=s.data.get(o.dataId).values,l;if(o.dtype===`string`){if(!Array.isArray(c))throw Error(`String tensor's value was not an instance of Array`);l=gy(c)}else l=c;let u=n||o.dtype,d=t(l,u,i);return s.makeTensorInfo(o.shape,u,d)}}var AT=DT(e=>Math.ceil(e));function jT(e,t,n,r){let i=ae(n,S(t));if(r&&n!==`string`){let t=0;e.forEach(e=>{let n=S(e.shape);i.set(e.vals,t),t+=n})}else{let r=0;e.forEach(e=>{let a=n===`string`?gy(e.vals):e.vals,o=0;for(let n=0;n<e.shape[0];++n){let s=n*t[1]+r;for(let t=0;t<e.shape[1];++t)i[s+t]=a[o++]}r+=e.shape[1]})}return i}var MT=gT((e,t)=>+(e===t)),NT=DT(e=>Math.exp(e)),PT=DT(e=>Math.expm1(e)),FT=DT(e=>Math.floor(e)),IT=gT((e,t)=>Math.floor(e/t));function LT(e,t,n,r,i,a,o,s,c){let l=I([r,a],n);for(let n=0;n<r;n++){let r=[],u=0;for(let t=0;t<i;t++){let a=e[n*i+t];u+=a*o[t],r.push(a)}if(u<0||u>=c/a)throw Error(`Invalid indices: ${r} does not index into ${s}`);for(let e=0;e<a;e++)l.values[n*a+e]=t.get(...t.indexToLoc(u*a+e))}return l}function RT(e,t,n){let r=I(n,e.dtype);for(let n=0;n<r.size;++n){let i=r.indexToLoc(n).slice(),a=i[0],o=i[2],s=t.locToIndex([a,o]);i[2]=t.values[s];let c=e.locToIndex(i);0<=c&&c<e.values.length&&(r.values[n]=e.values[c])}return r}var zT=gT((e,t)=>+(e>t)),BT=gT((e,t)=>+(e>=t)),VT=gT((e,t)=>+(e<t)),HT=gT((e,t)=>+(e<=t));function UT(e,t,n){let r=(t-e)/(n-1),i=xe(n,`float32`);i[0]=e;for(let e=1;e<i.length;e++)i[e]=i[e-1]+r;return i}var WT=DT(e=>Math.log(e));function GT(e,t,n,r){let i=ie(r,S(n));for(let n=0;n<i.length;++n){let r=n*t,a=e[r];for(let n=0;n<t;++n){let t=e[r+n];(Number.isNaN(t)||t>a)&&(a=t)}i[n]=a}return i}var KT=gT(((e,t)=>Math.max(e,t))),qT=gT(((e,t)=>Math.min(e,t))),JT=gT(((e,t)=>e*t));function YT(e,t,n){return JT([],t,$r(-1,n),e,n)}var XT=gT(((e,t)=>e===t?0:1));function ZT(e,t,n,r,i){let a=t.length,o=S(t),s=_e(t),c=_e(i),l=ie(n,S(i));for(let t=0;t<o;++t){let n=we(t,a,s),i=Array(n.length);for(let e=0;e<i.length;e++)i[e]=n[r[e]];let o=Ce(i,a,c);l[o]=e[t]}return l}function QT(e,t,n,r){let[i,a]=Tl(e,r),o=Fi(t,`int32`),s=xe(S(i),o),c=S(a);for(let e=0;e<s.length;++e){let t=e*c,r=1;for(let e=0;e<c;++e)r*=n[t+e];s[e]=r}return{outVals:s,outShape:i,outDtype:o}}function $T(e,t,n){e.forEach((e,r)=>{if(e<0||e>=n){let i=we(r,t.length,_e(t)).join(`,`);throw Error(`indices[${i}] = ${e} is not in [0, ${n})`)}})}function eE(e,t){for(let n=0;n<e.length;++n){let r=e[n],i=n===e.length-1?t:e[n+1].length;if(r.length===0)throw Error(`Ragged splits may not be empty`);if(r[0]<0)throw Error(`Ragged splits must be non-negative`);if(r[r.length-1]>i)throw Error(`Ragged splits must not point past values`);for(let e=1;e<r.length;++e)if(r[e-1]>r[e])throw Error(`Ragged splits must be sorted in ascending order`)}}function tE(e,t,n,r){let i=[],a=0,o=t.length-1+n.length,s=Array(o).fill(null).map(()=>[0]);eE(n,r);let c=1;for(let e=0;e<t.length-1;++e){c*=t[e];let n=t[e+1];for(let t=1;t<c+1;++t)s[e].push(t*n)}for(let r=0;r<e.length;++r){let o=e[r],c=e[r]+1;for(let e=0;e<n.length;++e){let r=n[e],i=e+t.length-1;if(i>=0){let e=s[i],t=e[e.length-1]-r[o];for(let e=o;e<c;++e)s[i].push(r[e+1]+t)}o=r[o],c=r[c]}c!==o&&(i.push([o,c]),a+=c-o)}return{outSplits:s,valueSlices:i,numValues:a}}function nE(e){let t=[];for(let n=0;n<e.length;++n){let r=e[n].length,i=ae(`int32`,r);t.push(i),e[n].forEach((e,t)=>i[t]=e)}return t}function rE(e,t){let n=e.slice(0,t);for(;n.length<t;)n.push(1);for(let r=t;r<e.length;r++)n[t-1]*=e[r];return n}function iE(e,t,n,r,i,a){let o=rE(t,2)[1],s=rE(a,2)[1],c=0;for(let t of n)for(let n=t[0];n<t[1];++n){for(let t=0;t<r;++t)i[c*s+t]=e[n*o+t];++c}}function aE(e,t,n,r,i){let a=t.slice();a[0]=i;let o=ae(n,S(a)),s=e.length;return iE(e,t,r,s===0?0:s/t[0],o,a),[o,a]}function oE(e,t,n,r,i,a,o,s){if(e.length===0)throw Error(`paramsNestedSplits must be non empty`);if(t[0].length===0)throw Error(`Split tensors must not be scalars`);if($T(a,o,t[0][0]-1),r.length===0)throw Error(`params.rank must be nonzero`);let c=r[0],{outSplits:l,valueSlices:u,numValues:d}=tE(a,o,e,c),f=nE(l),p=aE(n,r,i,u,d);return[f,p[0],p[1]]}var sE=2147483647;function cE(e,t,n,r,i,a,o){if(t.length>1)throw Error(`starts must be a scalar or vector`);if(i.length>1)throw Error(`limits must be a scalar or vector`);if(o.length>1)throw Error(`deltas must be a scalar or vector`);let s=t.length===0,c=i.length===0,l=o.length===0,u=[];s||u.push(t[0]),c||u.push(i[0]),l||u.push(o[0]);for(let e=1;e<u.length;++e)if(u[e]!==u[e-1])throw Error(`starts, limits, and deltas must have the same shape`);let d=u.length===0?1:u[0],f=ae(`int32`,d+1);f[0]=0;for(let t=0;t<d;++t){let n=s?e[0]:e[t],i=c?r[0]:r[t],o=l?a[0]:a[t];if(o===0)throw Error(`Requires delta != 0`);let u;if(o>0&&i<n||o<0&&i>n)u=0;else if(u=Math.ceil(Math.abs((i-n)/o)),u>sE)throw Error(`Requires ((limit - start) / delta) <= ${sE}`);f[t+1]=f[t]+u}let p=f[d],m=ae(n,p),h=0;for(let t=0;t<d;++t){let n=f[t+1]-f[t],r=s?e[0]:e[t],i=l?a[0]:a[t];for(let e=0;e<n;++e)m[h++]=r,r+=i}return[f,m]}var lE=mv,uE=class e{constructor(e,t,n,r,i,a,o,s,c,l){this.shape=e,this.shapeShape=t,this.values=n,this.valuesShape=r,this.valuesDType=i,this.defaultValue=a,this.defaultValueShape=o,this.rowPartitionValues=s,this.rowPartitionValuesShapes=c,this.rowPartitionTypes=gv(l),this.raggedRank=_v(this.rowPartitionTypes)}getRowPartitionTypeByDimension(e){return this.rowPartitionTypes[0]===lE.FIRST_DIM_SIZE?this.rowPartitionTypes[e+1]:this.rowPartitionTypes[e]}getRowPartitionTensor(e){return this.rowPartitionTypes[0]===lE.FIRST_DIM_SIZE?this.rowPartitionValues[e+1]:this.rowPartitionValues[e]}getMaxWidth(t){let n=this.getRowPartitionTensor(t-1);switch(this.getRowPartitionTypeByDimension(t-1)){case lE.VALUE_ROWIDS:return e.getMaxWidthValueRowID(n);case lE.ROW_SPLITS:return e.getMaxWidthRowSplit(n);default:throw Error(`Cannot handle partition type ${lE[this.getRowPartitionTypeByDimension(t-1)]}`)}}static getMaxWidthRowSplit(e){let t=e.length;if(t===0||t===1)return 0;let n=0;for(let r=0;r<t-1;++r){let t=e[r+1]-e[r];t>n&&(n=t)}return n}static getMaxWidthValueRowID(e){let t=e.length;if(t===0)return 0;let n=0,r=e[0],i=0;for(let a=1;a<t;++a){let t=e[a];t!==r&&(r=t,i=Math.max(a-n,i),n=a)}return Math.max(t-n,i)}tensorShapeFromTensor(e,t,n=!0){if(t.length===0){if(e[0]===-1)return[];throw Error(`The only valid scalar shape tensor is the fully unknown shape specified as -1.`)}return fE(e,n)}calculateOutputSize(e){let t=this.valuesShape,n=this.defaultValueShape;vv(n,t);let r=this.tensorShapeFromTensor(this.shape,this.shapeShape),i=hv(this.raggedRank,r,t);i[0]<0&&(i[0]=e);for(let e=1;e<=this.raggedRank;++e)i[e]<0&&(i[e]=this.getMaxWidth(e));return i}calculateFirstParentOutputIndex(e,t,n){let r=Math.min(e,n),i=[],a=0;for(let e=0;e<r;++e,a+=t)i.push(a);for(let t=r;t<e;++t)i.push(-1);return y(i.length===e,()=>`Final length of result must be equal to firstDimension.`),i}calculateOutputIndexRowSplit(e,t,n,r){let i=e.length,a=[];for(let o=0;o<i-1;++o){let i=e[o+1]-e[o],s=Math.min(r,i),c=t[o];c===-1&&(s=0);for(let e=0;e<s;++e)a.push(c),c+=n;for(let e=0;e<i-s;++e)a.push(-1)}if(i>0&&a.length!==e[i-1])throw Error(`Invalid row split size.`);return a}calculateOutputIndexValueRowID(e,t,n,r){let i=e.length,a=[];if(i===0)return[];let o=0,s=e[0];if(s>=t.length)throw Error(`Got currentValueRowId=${s}, which is not less than ${t.length}`);let c=t[s];a.push(c);for(let l=1;l<i;++l){let i=e[l];if(i===s)c>=0&&(++o,o<r?c+=n:c=-1);else{if(o=0,s=i,i>=t.length)throw Error(`Got nextValueRowId=${i} which is not less than ${t.length}`);c=t[i]}a.push(c)}if(a.length!==e.length)throw Error(`Invalid row ids.`);return a}calculateOutputIndex(e,t,n,r){let i=this.getRowPartitionTensor(e),a=this.getRowPartitionTypeByDimension(e);switch(a){case lE.VALUE_ROWIDS:return this.calculateOutputIndexValueRowID(i,t,n,r);case lE.ROW_SPLITS:if(i.length-1>t.length)throw Error(`Row partition size is greater than output size: ${i.length-1} > ${t.length}`);return this.calculateOutputIndexRowSplit(i,t,n,r);default:throw Error(`Unsupported partition type: ${lE[a]}`)}}getFirstDimensionSize(){let e=this.rowPartitionValues[0];if(this.rowPartitionTypes.length===0)throw Error(`No row_partition_types given.`);let t=this.rowPartitionTypes[0];switch(t){case lE.FIRST_DIM_SIZE:return e[0];case lE.VALUE_ROWIDS:throw Error(`Cannot handle VALUE_ROWIDS in first dimension.`);case lE.ROW_SPLITS:return this.rowPartitionValuesShapes[0][0]-1;default:throw Error(`Cannot handle type ${lE[t]}`)}}compute(){if(this.rowPartitionValues[0].length<=0)throw Error(`Invalid first partition input. Tensor requires at least one element.`);let e=this.getFirstDimensionSize(),t=this.calculateOutputSize(e),n=Array(this.raggedRank+1);n[n.length-1]=1;for(let e=n.length-2;e>=0;--e)n[e]=n[e+1]*t[e+1];let r=fE(t,!1),i=ae(this.valuesDType,S(r));if(n[0]*t[0]>0){let a=this.calculateFirstParentOutputIndex(e,n[0],t[0]);for(let e=1;e<=this.raggedRank;++e)a=this.calculateOutputIndex(e-1,a,n[e],t[e]);this.setOutput(this.raggedRank,a,i,r)}return[r,i]}setOutput(e,t,n,r){if(n.length===0)return;let i=this.values,a=n,o=r.slice();o=o.slice(e+1);let s=S(o),c=t.length,l=this.defaultValue;if(l.length!==s&&l.length!==1){let e=this.defaultValueShape;P(()=>{l=dc(V(l,e),o).dataSync()})}let u=0,d=0,f=0;for(let e=0;e<=c;++e){let r=e<c?t[e]:-1;if(r===f){++f;continue}if(d<f){let e=i.subarray(u*s);dE(a.subarray(d*s),e,(f-d)*s)}if(e>=c){let e=n.length;r=Math.floor(e/s)}if(r>f)if(this.defaultValue.length===1)a.subarray(f*s,r*s).fill(this.defaultValue[0]),f=r;else for(;r>f;)dE(a.slice(f*s),l,s),++f;r<0?(u=e+1,d=f):(u=e,d=f,f=d+1)}}};function dE(e,t,n){for(let r=0;r<n;r++)e[r]=t[r]}function fE(e,t){let n=[];for(let r of e){if(r<0){if(!t)throw Error(`Dimension ${r} must be >= 0`);if(r<-1)throw Error(`Dimension ${r} must be >= -1`);r=-1}n.push(r)}return n}function pE(e,t,n,r,i,a,o,s,c,l){return new uE(e,t,n,r,i,a,o,s,c,l).compute()}function mE(e,t,n,r){if(e===t||e<t&&n<0||t<e&&n>1)return xe(0,r);let i=xe(Math.abs(Math.ceil((t-e)/n)),r);t<e&&n===1&&(n=-1),i[0]=e;for(let e=1;e<i.length;e++)i[e]=i[e-1]+n;return i}var hE=DT(e=>1/Math.sqrt(e));function gE(e,t,n,r,i,a,o,s,c,l){let u=[r/i,i],d=e.values,f=t.values;if(r===0)return I(n,t.dtype);let p=c instanceof xi?c:I(u,t.dtype);typeof c==`string`||typeof c==`number`?p.values.fill(c):typeof c==`boolean`&&p.values.fill(+c);for(let e=0;e<a;e++){let a=[],c=0;for(let t=0;t<o;t++){let n=d[e*o+t];a.push(n),c+=n*s[t]}if(c<0||c>=r/i)throw Error(`Invalid indices: ${a} does not index into ${n}`);for(let n=0;n<i;n++)l?p.values[c*i+n]+=f[e*i+n]:p.values[c*i+n]=t.rank===0?f[0]:f[e*i+n]}return p}var _E=DT(e=>1/(1+Math.exp(-e)));OT(qn,e=>1/(1+Math.exp(-e)));function vE(e,t,n,r,i){let a=iv(r,t,n),o=S(n),s=_e(r);if(a){let n=av(t,s);return i===`string`?e.slice(n,n+o):e.subarray(n,n+o)}let c=I(r,i,i===`string`?gy(e):e),l=I(n,i);for(let e=0;e<l.size;++e){let n=l.indexToLoc(e),r=n.map((e,n)=>e+t[n]);l.set(c.get(...r),...n)}return i===`string`?_y(l.values):l.values}function yE(e,t,n,r,i,a,o){let s=t[0],c=a[0],l=Array(c),u=Array(s),d=t[1];if(c===0){if(s!==0)throw Error($v(s));let e=ae(n,0),t=ae(i,0);return[e,[0,d],t,l,u]}let f=!0,p=0,m=Array(c).fill(0);for(let t=0;t<s;++t){let n=e[t*d];if(n<0)throw Error(ey(t,n));if(n>=c)throw Error(ty(t,n,c));++m[n],f&&=n>=p,p=n}let h=!0;for(let e=0;e<c;++e){let t=m[e]===0;l[e]=t,h&&=!t,m[e]=Math.max(m[e],1),e>0&&(m[e]+=m[e-1])}if(h&&f){let t=e,n=r;for(let e=0;e<s;++e)u[e]=e;return[t,[s,d],n,l,u]}else{let t=m[c-1],a=ae(n,t*d),f=ae(i,t),p=Array(c).fill(0);for(let t=0;t<s;++t){let n=e[t*d],i=p[n],o=(n===0?0:m[n-1])+i;p[n]++;for(let n=0;n<d;++n)a[o*d+n]=e[t*d+n];f[o]=r[t],u[t]=o}for(let e=0;e<c;++e)if(p[e]===0){let t=e===0?0:m[e-1];a[t*d+0]=e;for(let e=1;e<d;++e)a[t*d+e]=0;f[t]=o}return[a,[t,d],f,l,u]}}function bE(e,t,n,r,i){let a=S(r),o=t[0],s=i.length,c=[],l=1,u=-1;for(let e=0;e<s;++e){let t=i[e];if(t===-1){if(u!==-1)throw Error(ny(u,e));u=e,c.push(1)}else{if(t<0)throw Error(ry(e,t));l*=t,c.push(t)}}if(u!==-1){if(l<=0)throw Error(iy());let e=Math.trunc(a/l);if(l*e!==a)throw Error(ay(r,c));c[u]=e}if(S(c)!==a)throw Error(oy(r,c));let d=r.length,f=[];if(d>0){f[d-1]=1;for(let e=d-2;e>=0;--e)f[e]=f[e+1]*r[e+1]}let p=[];if(s>0){p[s-1]=1;for(let e=s-2;e>=0;--e)p[e]=p[e+1]*c[e+1]}let m=ae(n,o*s);for(let t=0;t<o;++t){let n=0;for(let r=0;r<d;++r)n+=e[t*d+r]*f[r];for(let e=0;e<s;++e)m[t*s+e]=Math.trunc(n/p[e]),n%=p[e]}return[m,[o,s],c]}function xE(e,t,n,r,i,a=!1,o=0){let s=r.length,c=[t[0],e.length/t[0]],l=c[1],u=s>0?i[s-1]+1:0;if(u<0)throw Error(sy());let d=t.slice();d[0]=u;let f=ae(n,d.reduce((e,t)=>e*t,1));if(s===0)return u>0&&f.fill(o),[f,d];if(u<=0)throw Error(sy());let p=0,m=1,h=0,g=i[p];for(;;){let t=0;if(m<s){if(t=i[m],g===t){++m;continue}if(g>=t)throw Error(cy())}if(g<0||g>=u)throw Error(ly(g,u));g>h&&f.fill(o,h*l,g*l);for(let t=p;t<m;++t){let n=r[t];if(n<0||n>=c[0])throw Error(uy(t,r[t],c[0]));for(let t=0;t<l;t++)f[g*l+t]+=e[n*l+t]}if(a)for(let e=0;e<l;e++)f[g*l+e]/=m-p;if(p=m,++m,h=g+1,g=t,m>s)break}return h<u&&f.fill(o,h*l,u*l),[f,d]}var SE=DT(e=>Math.sqrt(e));OT(Yn,e=>Math.sqrt(e));var CE=gT(((e,t)=>{let n=e-t;return n*n})),wE=DT((e,t)=>{let{pattern:n,replaceGlobal:r,rewrite:i}=t;return e.replace(new RegExp(n,r?`g`:``),i)});function TE(e,t,n,r){let i=I(e,t.dtype);for(let e=0;e<i.size;e++){let a=i.indexToLoc(e),o=Array(a.length);for(let e=0;e<o.length;e++)o[e]=a[e]*n[e]+r[e];i.set(t.get(...o),...a)}return i}var EE=class{constructor(e,t,n,r,i,a){this.separator=ri(e),this.nGramWidths=t,this.leftPad=ri(n),this.rightPad=ri(r),this.padWidth=i,this.preserveShort=a}getPadWidth(e){return Math.min(this.padWidth<0?e-1:this.padWidth,e-1)}getNumNGrams(e,t){let n=this.getPadWidth(t);return Math.max(0,e+2*n-t+1)}createNGrams(e,t,n,r,i,a){for(let o=0;o<i;++o){let s=this.getPadWidth(a),c=Math.max(0,s-o),l=Math.max(0,s-(i-(o+1))),u=a-(c+l),d=t+(c>0?0:o-s),f=0;f+=c*this.leftPad.length;for(let t=0;t<u;++t)f+=e[d+t].length;f+=l*this.rightPad.length;let p=c+l+u-1;f+=p*this.separator.length,n[r+o]=new Uint8Array(f);let m=n[r+o],h=0,g=e=>e.forEach(e=>m[h++]=e);for(let e=0;e<c;++e)g(this.leftPad),g(this.separator);for(let t=0;t<u-1;++t)g(e[d+t]),g(this.separator);if(u>0){g(e[d+u-1]);for(let e=0;e<l;++e)g(this.separator),g(this.rightPad)}else{for(let e=0;e<l-1;++e)g(this.rightPad),g(this.separator);g(this.rightPad)}}}compute(e,t){let n=e.length,r=t.length;if(r>0){let e=t[0];if(e!==0)throw Error(`First split value must be 0, got ${e}`);for(let i=1;i<r;++i){let r=t[i]>=e;if(r&&=t[i]<=n,!r)throw Error(`Invalid split value ${t[i]}, must be in [${e}, ${n}]`);e=t[i]}if(e!==n)throw Error(`Last split value must be data size. Expected ${n}, got ${e}`)}let i=r-1,a=ae(`int32`,r);if(n===0||r===0){let e=Array(n);for(let e=0;e<=i;++e)a[e]=0;return[e,a]}a[0]=0;for(let e=1;e<=i;++e){let n=t[e]-t[e-1],r=0;this.nGramWidths.forEach(e=>{r+=this.getNumNGrams(n,e)}),this.preserveShort&&n>0&&r===0&&(r=1),a[e]=a[e-1]+r}let o=Array(a[i]);for(let n=0;n<i;++n){let r=t[n],i=a[n];if(this.nGramWidths.forEach(a=>{let s=t[n+1]-t[n],c=this.getNumNGrams(s,a);this.createNGrams(e,r,o,i,c,a),i+=c}),this.preserveShort&&i===a[n]){let a=t[n+1]-t[n];if(a===0)continue;let s=a+2*this.padWidth;this.createNGrams(e,r,o,i,1,s)}}return[o,a]}};function DE(e,t,n,r,i,a,o,s){return new EE(n,r,i,a,o,s).compute(e,t)}function OE(e,t,n,r){if(!e.length)return;if(t.length===0){for(let t=0;t<e.length;++t)r.push(e.subarray(t,t+1));return}if(t.length===1){let i=t[0],a=e.indexOf(i);for(;a!==-1;){let t=e.subarray(0,a);(!n||t.length!==0)&&r.push(t),e=e.subarray(a+1),a=e.indexOf(i)}(!n||e.length!==0)&&r.push(e);return}let i=0;for(let a=0;a<e.length+1;a++)if(a===e.length||t.indexOf(e[a])!==-1){let t=e.subarray(i,a);(!n||t.length!==0)&&r.push(t),i=a+1}}function kE(e,t,n){let r=e.length,i=[],a=0,o=0,s=Array(r);for(let c=0;c<r;++c){let r=i.length;OE(e[c],t,n,i);let l=i.length-r;s[c]=l,a+=l,o=Math.max(o,l)}let c=ae(`int32`,a*2),l=Array(a),u=[r,o],d=0;for(let e=0;e<r;++e)for(let t=0;t<s[e];++t)c[d*2]=e,c[d*2+1]=t,l[d]=i[d],++d;return[c,l,u]}function AE(e,t){let n=ae(`int32`,e.length);for(let r=0;r<e.length;++r)n[r]=Qr(e[r]).modulo(t).getLowBitsUnsigned();return n}var jE=gT(((e,t)=>e-t));function ME(e,t){let n=Array(e.rank);for(let r=0;r<n.length;r++)n[r]=e.shape[r]*t[r];let r=I(n,e.dtype);for(let t=0;t<r.values.length;++t){let n=r.indexToLoc(t),i=Array(e.rank);for(let t=0;t<i.length;t++)i[t]=n[t]%e.shape[t];let a=e.locToIndex(i);r.values[t]=e.values[a]}return r}var NE=(e,t)=>{let n=t.value-e.value;return n===0?e.index-t.index:n};function PE(e,t,n=0,r=e.length-1){for(;r>n;){if(r-n>600){let i=r-n+1,a=t-n+1,o=Math.log(i),s=.5*Math.exp(2*o/3),c=.5*Math.sqrt(o*s*(i-s)/i)*Math.sign(a-i/2);PE(e,t,Math.max(n,Math.floor(t-a*s/i+c)),Math.min(r,Math.floor(t+(i-a)*s/i+c)))}let i=e[t],a=n,o=r;for(_(e,n,t),NE(e[r],i)>0&&_(e,n,r);a<o;){for(_(e,a,o),a++,o--;NE(e[a],i)<0;)a+=1;for(;NE(e[o],i)>0;)--o}NE(e[n],i)===0?_(e,n,o):(o+=1,_(e,o,r)),o<=t&&(n=o+1),t<=o&&(r=o-1)}}function FE(e,t,n,r,i){let a=t[t.length-1],[o,s]=[e.length/a,a],c=ie(n,o*r),l=ie(`int32`,o*r);for(let t=0;t<o;t++){let n=t*s,a=e.subarray(n,n+s),o=Array(a.length);a.forEach((e,t)=>o[t]={value:e,index:t}),r<o.length&&(PE(o,r),o=o.slice(0,r)),i&&o.sort(NE);let u=t*r,d=c.subarray(u,u+r),f=l.subarray(u,u+r);for(let e=0;e<r;e++)d[e]=o[e].value,f[e]=o[e].index}let u=t.slice();return u[u.length-1]=r,[I(u,n,c),I(u,`int32`,l)]}function IE(e,t,n,r){let i=D(t,n)[0],a=[1,n[0],1];for(let e=0;e<i;e++)a[0]*=n[e];a[1]=n[i];for(let e=i+1;e<n.length;e++)a[2]*=n[e];let o=new Map,s=new Int32Array(n[i]),c=new xi(a,r,e),l=[],u=a[0]===1&&a[2]===1;for(let t=0;t<n[i];t++){let n;if(u)n=e[t].toString();else{let e=[];for(let n=0;n<a[0];n++)for(let r=0;r<a[2];r++)e.push(c.get(n,t,r));n=e.join(`,`)}let r=o.get(n);if(r!=null)s[t]=r;else{let e=o.size;o.set(n,e),s[t]=e,l.push(t)}}let d=a.slice();d[1]=o.size;let f=new xi(d,r);l.forEach((e,t)=>{for(let n=0;n<a[0];n++)for(let r=0;r<a[2];r++)f.set(c.get(n,e,r),n,t,r)});let p=n.slice();return p[i]=d[1],{outputValues:f.values,outputShape:p,indices:s}}var{addImpl:LE,bincountImpl:RE,bincountReduceImpl:zE,bitwiseAndImpl:BE,castImpl:VE,ceilImpl:HE,concatImpl:UE,equalImpl:WE,expImpl:GE,expm1Impl:KE,floorImpl:qE,gatherNdImpl:JE,gatherV2Impl:YE,greaterImpl:XE,greaterEqualImpl:ZE,lessImpl:QE,lessEqualImpl:$E,linSpaceImpl:eD,logImpl:tD,maxImpl:nD,maximumImpl:rD,minimumImpl:iD,multiplyImpl:aD,negImpl:oD,notEqualImpl:sD,prodImpl:cD,raggedGatherImpl:lD,raggedRangeImpl:uD,raggedTensorToTensorImpl:dD,rangeImpl:fD,rsqrtImpl:pD,scatterImpl:mD,sigmoidImpl:hD,simpleAbsImpl:gD,sliceImpl:_D,sparseFillEmptyRowsImpl:vD,sparseReshapeImpl:yD,sparseSegmentReductionImpl:bD,sqrtImpl:xD,staticRegexReplaceImpl:SD,stridedSliceImpl:CD,stringNGramsImpl:wD,stringSplitImpl:TD,stringToHashBucketFastImpl:ED,subImpl:DD,tileImpl:OD,topKImpl:kD,transposeImpl:AD,uniqueImpl:jD}=s({addImpl:()=>CT,bincountImpl:()=>wT,bincountReduceImpl:()=>TT,bitwiseAndImpl:()=>ET,castImpl:()=>xT,ceilImpl:()=>AT,concatImpl:()=>jT,equalImpl:()=>MT,expImpl:()=>NT,expm1Impl:()=>PT,floorDivImpl:()=>IT,floorImpl:()=>FT,gatherNdImpl:()=>LT,gatherV2Impl:()=>RT,greaterEqualImpl:()=>BT,greaterImpl:()=>zT,lessEqualImpl:()=>HT,lessImpl:()=>VT,linSpaceImpl:()=>UT,logImpl:()=>WT,maxImpl:()=>GT,maximumImpl:()=>KT,minimumImpl:()=>qT,multiplyImpl:()=>JT,negImpl:()=>YT,notEqualImpl:()=>XT,prodImpl:()=>QT,raggedGatherImpl:()=>oE,raggedRangeImpl:()=>cE,raggedTensorToTensorImpl:()=>pE,rangeImpl:()=>mE,rsqrtImpl:()=>hE,scatterImpl:()=>gE,sigmoidImpl:()=>_E,simpleAbsImpl:()=>hT,sliceImpl:()=>vE,sparseFillEmptyRowsImpl:()=>yE,sparseReshapeImpl:()=>bE,sparseSegmentReductionImpl:()=>xE,sqrtImpl:()=>SE,squaredDifferenceImpl:()=>CE,staticRegexReplaceImpl:()=>wE,stridedSliceImpl:()=>TE,stringNGramsImpl:()=>DE,stringSplitImpl:()=>kE,stringToHashBucketFastImpl:()=>AE,subImpl:()=>jE,tileImpl:()=>ME,topKImpl:()=>FE,transposeImpl:()=>ZT,uniqueImpl:()=>IE});function MD(e,t){return[`x`,`y`,`z`,`w`,`u`,`v`].slice(0,t).map(t=>`${e}.${t}`)}function ND(e,t){return t===1?[e]:MD(e,t)}function PD(e,t){if(e===1)return`rc`;let n=``;for(let r=0;r<e;r++)n+=t[r],r<e-1&&(n+=`,`);return n}var FD=class{constructor(e){if(this.variableNames=[`A`],this.packedInputs=!1,this.packedOutput=!0,this.outputShape=e,this.rank=e.length,this.enableShapeUniforms=Iw(this.outputShape.length),this.rank===0)this.userCode=`
        void main() {
          setOutput(vec4(getA(), 0., 0., 0.));
        }
      `;else{let e=ND(`rc`,this.rank),t=Z(this.rank),n=this.getOutOfBoundsCondition(e),r=this.getSetup(e),i=this.getOutput(e);this.userCode=`
        void main() {
          ${t} rc = getOutputCoords();

          if(${n}) {
            setOutput(vec4(0));
          } else {
            ${r}

            setOutput(vec4(${i}));
          }
        }
      `}}getSourceCoordsArr(e){let t=[];for(let n=0;n<=1;n++)for(let r=0;r<=1;r++){let i=`${n===0?`r`:`rp1`}, ${r===0?`c`:`cp1`}`;for(let t=2;t<this.rank;t++)i=`${e[e.length-1-t]},`+i;t.push(i)}return t}getOutOfBoundsCondition(e){if(this.rank===1)return`rc > ${this.enableShapeUniforms?`outShape`:this.outputShape[0]}`;let t=``;for(let n=this.rank-2;n<this.rank;n++)t+=`${e[n]} >= ${this.enableShapeUniforms?`outShape[${n}]`:this.outputShape[n]}`,n<this.rank-1&&(t+=`||`);return t}getSetup(e){if(this.rank===1)return``;let t=e.slice(-2),n=this.enableShapeUniforms?`outShape[${this.rank} - 1]`:this.outputShape[this.rank-1],r=this.enableShapeUniforms?`outShape[${this.rank} - 2]`:this.outputShape[this.rank-2];return`
      int r = ${t[0]};
      int c = ${t[1]};
      int rp1 = r + 1;
      int cp1 = c + 1;

      bool cEdge = cp1 >= ${n};
      bool rEdge = rp1 >= ${r};
    `}getOutput(e){let t=this.getSourceCoordsArr(e);return this.rank===1?`getA(rc), (rc + 1 >= ${this.enableShapeUniforms?`outShape`:this.outputShape[0]} ? 0. : getA(rc + 1)), 0, 0`:`getA(${t[0]}),
            cEdge ? 0. : getA(${t[1]}),
            rEdge ? 0. : getA(${t[2]}),
            rEdge || cEdge ? 0. : getA(${t[3]})`}},ID=class{constructor(e,t){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`inputShape`,type:`ivec3`}],this.outputShape=e,this.enableShapeUniforms=Iw(this.outputShape.length);let n=``;for(let e=0;e<4;e++){let t=`thisRC = rc;`;e%2==1&&(t+=`thisRC.z += 1;`),e>1&&(t+=`thisRC.y += 1;`),n+=`
        ${t}
        ${e>0?`if(thisRC.y < rows && thisRC.z < cols){`:``}
          int flatIndex = getFlatIndex(thisRC);

          ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
          vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));

          result[${e}] =
            getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);
        ${e>0?`}`:``}
      `}this.userCode=`
      ${LD(t,this.enableShapeUniforms)}
      ${this.enableShapeUniforms?RC():LC(e)}

      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0.);

        ivec3 thisRC;
        int rows = ${this.enableShapeUniforms?`outShape[1]`:e[1]};
        int cols = ${this.enableShapeUniforms?`outShape[2]`:e[2]};

        ${n}

        setOutput(result);
      }
    `}};function LD(e,t){return`
    ivec3 inputCoordsFromReshapedOutCoords(int index) {
      ${t?IC([`r`,`c`,`d`],`inputShape`):NC([`r`,`c`,`d`],e)}
      return ivec3(r, c, d);
    }
  `}var RD=class{constructor(e){this.gpgpu=e,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0,this.freeTextures={},this.usedTextures={},this.logEnabled=!1}acquireTexture(e,t,n){let r=UD(t,n),i=WD(e,r,n);i in this.freeTextures||(this.freeTextures[i]=[]),i in this.usedTextures||(this.usedTextures[i]=[]);let a=BD(e,r,this.gpgpu.gl,this.gpgpu.textureConfig,n);if(this.freeTextures[i].length>0){this.numFreeTextures--,this.numUsedTextures++,this._numBytesFree-=a,this.log();let e=this.freeTextures[i].pop();return this.usedTextures[i].push(e),e}let o;return r===jS.PACKED_2X2_FLOAT32?o=this.gpgpu.createPackedMatrixTexture(e[0],e[1]):r===jS.PACKED_2X2_FLOAT16?o=this.gpgpu.createFloat16PackedMatrixTexture(e[0],e[1]):r===jS.UNPACKED_FLOAT32?o=this.gpgpu.createFloat32MatrixTexture(e[0],e[1]):r===jS.UNPACKED_FLOAT16?o=this.gpgpu.createFloat16MatrixTexture(e[0],e[1]):r===jS.PACKED_4X1_UNSIGNED_BYTE&&(o=this.gpgpu.createUnsignedBytesMatrixTexture(e[0],e[1])),this.usedTextures[i].push(o),this.numUsedTextures++,this._numBytesAllocated+=a,this.log(),o}releaseTexture(e,t,n,r){if(this.freeTextures==null)return;let i=UD(n,r),a=WD(t,i,r);a in this.freeTextures||(this.freeTextures[a]=[]);let o=BD(t,i,this.gpgpu.gl,this.gpgpu.textureConfig,r),s=O().getNumber(`WEBGL_DELETE_TEXTURE_THRESHOLD`);s!==-1&&this._numBytesAllocated>s?(this.gpgpu.deleteMatrixTexture(e.texture),this._numBytesAllocated-=o):(this.freeTextures[a].push(e),this.numFreeTextures++,this._numBytesFree+=o),this.numUsedTextures--;let c=this.usedTextures[a],l=c&&c.indexOf(e);if(l==null||l<0)throw Error(`Cannot release a texture that was never provided by this texture manager`);c[l]=c[c.length-1],c.pop(),this.log()}log(){if(!this.logEnabled)return;let e=this.numFreeTextures+this.numUsedTextures;console.log(`Free/Used`,`${this.numFreeTextures} / ${this.numUsedTextures}`,`(${e})`);let t=this._numBytesFree/this._numBytesAllocated;console.log(`Bytes allocated: ${this._numBytesAllocated}`),console.log(`Bytes unused: ${this._numBytesFree} (${Math.round(100*t)}%)`)}get numBytesAllocated(){return this._numBytesAllocated}get numBytesFree(){return this._numBytesFree}getNumUsedTextures(){return this.numUsedTextures}getNumFreeTextures(){return this.numFreeTextures}dispose(){if(this.freeTextures!=null){for(let e in this.freeTextures)this.freeTextures[e].forEach(e=>{this.gpgpu.deleteMatrixTexture(e.texture)});for(let e in this.usedTextures)this.usedTextures[e].forEach(e=>{this.gpgpu.deleteMatrixTexture(e.texture)});this.freeTextures=null,this.usedTextures=null,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0}}};function zD(e,t){let n=e;if(t===n.R32F)return 4;if(t===n.R16F)return 2;if(t===n.RGBA32F||t===e.RGBA)return 16;if(t===n.RGBA16F)return 8;if(t===n.RGBA8)return 4;throw Error(`Unknown internal format ${t}`)}function BD(e,t,n,r,i){let a=VD(t,r),o;if(i){let[t,n]=FS(e[0],e[1]);o=t*n}else{let[t,n]=MS(e[0],e[1]);o=t*n}let s=zD(n,a);return o*s}function VD(e,t){switch(e){case jS.PACKED_2X2_FLOAT32:return eT(t);case jS.PACKED_2X2_FLOAT16:return nT(t);case jS.UNPACKED_FLOAT32:return Jw(t);case jS.UNPACKED_FLOAT16:return Xw(t);case jS.PACKED_4X1_UNSIGNED_BYTE:return Qw(t);default:throw Error(`Unknown physical texture type ${e}`)}}function HD(e){return O().getBool(`WEBGL_RENDER_FLOAT32_ENABLED`)?e?jS.PACKED_2X2_FLOAT32:jS.UNPACKED_FLOAT32:e?jS.PACKED_2X2_FLOAT16:jS.UNPACKED_FLOAT16}function UD(e,t){if(e===AS.UPLOAD)return jS.PACKED_2X2_FLOAT32;if(e===AS.RENDER||e==null)return HD(t);if(e===AS.DOWNLOAD||e===AS.PIXELS)return jS.PACKED_4X1_UNSIGNED_BYTE;throw Error(`Unknown logical texture type ${e}`)}function WD(e,t,n){return`${e[0]}_${e[1]}_${t}_${n}`}var GD=class{constructor(e,t){this.variableNames=[`A`],this.outputShape=e,this.enableShapeUniforms=Iw(this.outputShape.length),this.userCode=`
      float unaryOperation(float x) {
        ${t}
      }

      void main() {
        float x = getAAtOutCoords();
        float y = unaryOperation(x);

        setOutput(y);
      }
    `}},KD=`if (isnan(x)) return x;`,qD=`return x;`,JD=`return abs(x);`,YD=`return (x >= 0.0) ? x : (exp(x) - 1.0);`,XD=KD+`
  return (x < 0.0) ? 0.0 : x;
`,ZD=KD+`
  return (x < 0.0) ? 0.0 : min(6.0, x);
`,QD=`return x;`,$D=`return 1.0 / (1.0 + exp(-1.0 * x));`,eO=`return x;`,tO=`
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`,nO=`
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,rO=`
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,iO=`return 1.0 / (1.0 + exp(-1.0 * x));`,aO=class{constructor(e,t){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.enableShapeUniforms=Iw(this.outputShape.length),this.userCode=`
      vec4 unaryOperation(vec4 x) {
        ${t}
      }

      void main() {
        vec4 x = getAAtOutCoords();
        vec4 y = unaryOperation(x);

        setOutput(y);
      }
    `}},oO=class{constructor(e){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!1,this.outputShape=e,this.enableShapeUniforms=Iw(this.outputShape.length);let t=e.length,n=ND(`rc`,t),r=Z(t),i=PD(t,n),a=n.slice(-2),o=t<=1?`rc`:`vec2(${a.join(`,`)})`;this.userCode=`
      void main() {
        ${r} rc = getOutputCoords();
        vec4 packedInput = getA(${i});

        setOutput(getChannel(packedInput, ${o}));
      }
    `}},sO=mm,cO=1e-7,lO=1e-4,uO={};function dO(e){return e in uO||(uO[e]={}),uO[e]}var fO=O().getNumber(`CPU_HANDOFF_SIZE_THRESHOLD`),pO=600;function mO(){return O().global.screen==null?1024:O().global.screen.height*O().global.screen.width*window.devicePixelRatio*pO/1024/1024}var hO=class e extends p{nextDataId(){return e.nextDataId++}constructor(e){if(super(),this.pendingRead=new WeakMap,this.pendingDisposal=new WeakSet,this.dataRefCount=new WeakMap,this.numBytesInGPU=0,this.uploadWaitMs=0,this.downloadWaitMs=0,this.lastGlFlushTime=0,this.warnedAboutMemory=!1,this.pendingDeletes=0,this.disposed=!1,!O().getBool(`HAS_WEBGL`))throw Error(`WebGL is not supported on this device`);let t;e==null?(t=new fT(ES(O().getNumber(`WEBGL_VERSION`))),this.binaryCache=dO(O().getNumber(`WEBGL_VERSION`)),this.gpgpuCreatedLocally=!0):(t=e instanceof fT?e:new fT(ES(O().getNumber(`WEBGL_VERSION`),e)),this.binaryCache={},this.gpgpuCreatedLocally=!1),this.gpgpu=t,this.canvas=this.gpgpu.gl.canvas,this.textureManager=new RD(this.gpgpu),this.numMBBeforeWarning=mO(),this.texData=new f(this,fa())}numDataIds(){return this.texData.numDataIds()-this.pendingDeletes}writeTexture(e,t,n,r,i,a){let o=this.makeTensorInfo(t,n),s=this.texData.get(o.dataId);s.isPacked=!1,s.texture={texture:e,texShape:[r,i]},s.texShape=[r,i];let c=new Hw(hC(t),!1,a),l=this.runWebGLProgram(c,[o],n,[[r,i]]);return l.shape=t,s.texture=null,this.disposeIntermediateTensorInfo(o),l.dataId}write(e,t,n){if((O().getBool(`WEBGL_CHECK_NUMERICAL_PROBLEMS`)||O().getBool(`DEBUG`))&&this.checkNumericalProblems(e),n===`complex64`&&e!=null)throw Error(`Cannot write to a complex64 dtype. Please use tf.complex(real, imag).`);let r={id:this.nextDataId()};return this.texData.set(r,{shape:t,dtype:n,values:e,usage:AS.UPLOAD,refCount:1}),r}refCount(e){return this.texData.has(e)?this.texData.get(e).refCount:0}incRef(e){let t=this.texData.get(e);t.refCount++}decRef(e){if(this.texData.has(e)){let t=this.texData.get(e);t.refCount--}}move(e,t,n,r,i){if(O().getBool(`DEBUG`)&&this.checkNumericalProblems(t),r===`complex64`)throw Error(`Cannot write to a complex64 dtype. Please use tf.complex(real, imag).`);this.texData.set(e,{shape:n,dtype:r,values:t,usage:AS.UPLOAD,refCount:i})}disposeIntermediateTensorInfo(e){this.disposeData(e.dataId)}readSync(e){let{values:t,dtype:n,complexTensorInfos:r,slice:i,shape:a,isPacked:o}=this.texData.get(e);if(i!=null){let t;t=o?new aO(a,QD):new GD(a,QD);let r=this.runWebGLProgram(t,[{dataId:e,shape:a,dtype:n}],n),i=this.readSync(r.dataId);return this.disposeIntermediateTensorInfo(r),i}if(t!=null)return this.convertAndCacheOnCPU(e);if(n===`string`)return t;let s=this.activeTimers!=null,c;s&&(c=ni());let l;return l=n===`complex64`?Pv(this.readSync(r.real.dataId),this.readSync(r.imag.dataId)):this.getValuesFromTexture(e),s&&(this.downloadWaitMs+=ni()-c),this.convertAndCacheOnCPU(e,l)}async read(e){if(this.pendingRead.has(e)){let t=this.pendingRead.get(e);return new Promise(e=>t.push(e))}let{values:t,shape:n,slice:r,dtype:i,complexTensorInfos:a,isPacked:o}=this.texData.get(e);if(r!=null){let t;t=o?new aO(n,QD):new GD(n,QD);let r=this.runWebGLProgram(t,[{dataId:e,shape:n,dtype:i}],i),a=this.read(r.dataId);return this.disposeIntermediateTensorInfo(r),a}if(t!=null)return this.convertAndCacheOnCPU(e);if(O().getBool(`DEBUG`)&&!O().getBool(`WEBGL_DOWNLOAD_FLOAT_ENABLED`)&&O().getNumber(`WEBGL_VERSION`)===2)throw Error(`tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and WEBGL_VERSION=2 not yet supported.`);let s=null,c;if(i!==`complex64`&&O().get(`WEBGL_BUFFER_SUPPORTED`)){c=this.decode(e);let t=this.texData.get(c.dataId);s=this.gpgpu.createBufferFromTexture(t.texture.texture,...PS(n))}this.pendingRead.set(e,[]),i!==`complex64`&&await this.gpgpu.createAndWaitForFence();let l;if(i===`complex64`){let e=await Promise.all([this.read(a.real.dataId),this.read(a.imag.dataId)]),t=e[0],n=e[1];l=Pv(t,n)}else if(s==null)l=this.getValuesFromTexture(e);else{let e=S(n);l=this.gpgpu.downloadFloat32MatrixFromBuffer(s,e)}if(c!=null&&this.disposeIntermediateTensorInfo(c),s!=null){let e=this.gpgpu.gl;Y(e,()=>e.deleteBuffer(s))}let u=this.convertAndCacheOnCPU(e,l),d=this.pendingRead.get(e);return this.pendingRead.delete(e),d.forEach(e=>e(u)),this.pendingDisposal.has(e)&&(this.pendingDisposal.delete(e),this.disposeData(e)&&fa().removeDataId(e,this),this.pendingDeletes--),u}readToGPU(e,t={}){let{values:n,shape:r,slice:i,dtype:a,isPacked:o,texture:s}=this.texData.get(e);if(a===`complex64`)throw Error(`Does not support reading texture for complex64 dtype.`);if(i!=null){let n;n=o?new aO(r,QD):new GD(r,QD);let i=this.runWebGLProgram(n,[{dataId:e,shape:r,dtype:a}],a),s=this.readToGPU(i,t);return this.disposeIntermediateTensorInfo(i),s}if(s==null)throw Error(n==null?`There is no data on GPU or CPU.`:`Data is not on GPU but on CPU.`);let c=this.decode(e,t.customTexShape),l=fa().makeTensorFromTensorInfo(c),u=this.texData.get(c.dataId);return Object.assign({tensorRef:l},u.texture)}bufferSync(e){let t=this.readSync(e.dataId);if(e.dtype===`string`)try{let n=t.map(e=>ii(e));return I(e.shape,e.dtype,n)}catch{throw Error(`Failed to decode encoded string bytes into utf-8`)}return I(e.shape,e.dtype,t)}checkNumericalProblems(e){if(e!=null)for(let t=0;t<e.length;t++){let n=e[t];if(!VS(n))throw O().getBool(`WEBGL_RENDER_FLOAT32_CAPABLE`)?Error(`The value ${n} cannot be represented with your current settings. Consider enabling float32 rendering: 'tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);'`):Error(`The value ${n} cannot be represented on this device.`)}}getValuesFromTexture(e){let{shape:t,dtype:n,isPacked:r}=this.texData.get(e),i=S(t);if(O().getBool(`WEBGL_DOWNLOAD_FLOAT_ENABLED`)){let n=this.decode(e),r=this.texData.get(n.dataId),a=this.gpgpu.downloadMatrixFromPackedTexture(r.texture.texture,...PS(t)).subarray(0,i);return this.disposeIntermediateTensorInfo(n),a}let a=O().getBool(`WEBGL_PACK`)&&r===!0,o=a?hC(t):t,s=a?new Bw(o):new zw(o),c=this.runWebGLProgram(s,[{shape:o,dtype:n,dataId:e}],`float32`),l=this.texData.get(c.dataId),u=this.gpgpu.downloadByteEncodedFloatMatrixFromOutputTexture(l.texture.texture,l.texShape[0],l.texShape[1]).subarray(0,i);return this.disposeIntermediateTensorInfo(c),u}timerAvailable(){return O().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE`)>0}time(e){let t=this.activeTimers,n=[],r=!1;this.programTimersStack==null?(this.programTimersStack=n,r=!0):this.activeTimers.push(n),this.activeTimers=n,e();let i=oi(this.activeTimers.map(e=>e.query)).filter(e=>e!=null),a=oi(this.activeTimers.map(e=>e.name)).filter(e=>e!=null);this.activeTimers=t,r&&(this.programTimersStack=null);let o={uploadWaitMs:this.uploadWaitMs,downloadWaitMs:this.downloadWaitMs,kernelMs:null,wallMs:null};return(async()=>{if(O().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE`)>0){let e=await Promise.all(i);o.kernelMs=v(e),o.getExtraProfileInfo=()=>e.map((e,t)=>({name:a[t],ms:e})).map(e=>`${e.name}: ${e.ms}`).join(`, `)}else o.kernelMs={error:`WebGL query timers are not supported in this environment.`};return this.uploadWaitMs=0,this.downloadWaitMs=0,o})()}memory(){return{unreliable:!1,numBytesInGPU:this.numBytesInGPU,numBytesInGPUAllocated:this.textureManager.numBytesAllocated,numBytesInGPUFree:this.textureManager.numBytesFree}}startTimer(){return O().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE`)>0?this.gpgpu.beginQuery():{startMs:ni(),endMs:null}}endTimer(e){return O().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE`)>0?(this.gpgpu.endQuery(),e):(e.endMs=ni(),e)}async getQueryTime(e){if(O().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE`)>0)return this.gpgpu.waitForQueryAndGetTime(e);let t=e;return t.endMs-t.startMs}disposeData(e,t=!1){if(this.pendingDisposal.has(e))return!1;if(!this.texData.has(e))return!0;if(t?this.texData.get(e).refCount=0:this.texData.get(e).refCount--,!t&&this.texData.get(e).refCount>0)return!1;if(this.pendingRead.has(e))return this.pendingDisposal.add(e),this.pendingDeletes++,!1;this.releaseGPUData(e);let{complexTensorInfos:n}=this.texData.get(e);return n!=null&&(this.disposeData(n.real.dataId,t),this.disposeData(n.imag.dataId,t)),this.texData.delete(e),!0}releaseGPUData(e){let{texture:t,dtype:n,texShape:r,usage:i,isPacked:a,slice:o}=this.texData.get(e),s=o&&o.origDataId||e,c=this.dataRefCount.get(s);c>1?this.dataRefCount.set(s,c-1):(this.dataRefCount.delete(s),t!=null&&(this.numBytesInGPU-=this.computeBytes(r,n),this.textureManager.releaseTexture(t,r,i,a)));let l=this.texData.get(e);l.texture=null,l.texShape=null,l.isPacked=!1,l.slice=null}getTexture(e){return this.uploadToGPU(e),this.texData.get(e).texture.texture}getDataInfo(e){return this.texData.get(e)}shouldExecuteOnCPU(e,t=fO){return O().getBool(`WEBGL_CPU_FORWARD`)&&e.every(e=>this.texData.get(e.dataId).texture==null&&S(e.shape)<t)}getGPGPUContext(){return this.gpgpu}where(e){Er(`tf.where() in webgl locks the UI thread. Call tf.whereAsync() instead`);let t=e.dataSync();return sO(e.shape,t)}packedUnaryOp(e,t,n){let r=new aO(e.shape,t),i=this.compileAndRun(r,[e],n);return fa().makeTensorFromTensorInfo(i)}abs(e){if(this.shouldExecuteOnCPU([e])&&e.dtype!==`complex64`){let t=gD(this.texData.get(e.dataId).values);return this.makeOutput(e.shape,e.dtype,t)}if(O().getBool(`WEBGL_PACK_UNARY_OPERATIONS`))return this.packedUnaryOp(e,JD,e.dtype);let t=new GD(e.shape,JD),n=this.compileAndRun(t,[e]);return fa().makeTensorFromTensorInfo(n)}makeTensorInfo(e,t,n){let r;if(t===`string`&&n!=null&&n.length>0&&de(n[0])){let i=n.map(e=>ri(e));r=this.write(i,e,t)}else r=this.write(n,e,t);return this.texData.get(r).usage=null,{dataId:r,shape:e,dtype:t}}makeOutput(e,t,n){return fa().makeTensorFromTensorInfo(this.makeTensorInfo(e,t,n),this)}unpackTensor(e){let t=new oO(e.shape);return this.runWebGLProgram(t,[e],e.dtype)}packTensor(e){let t=new FD(e.shape);return this.runWebGLProgram(t,[e],e.dtype,null,!0)}packedReshape(e,t){let n=[pC(e.shape),...mC(e.shape)],r={dtype:e.dtype,shape:n,dataId:e.dataId},i=new ID([pC(t),...mC(t)],n),a=[n],o=this.runWebGLProgram(i,[r],e.dtype,a,!0);return{dataId:o.dataId,shape:t,dtype:o.dtype}}decode(e,t){let{isPacked:n,shape:r,dtype:i}=this.texData.get(e);t!=null&&y(S(r)<=t[0]*t[1]*4,()=>`customTexShape is too small. Row * Column * 4 should be equal or larger than the size of the tensor data.`);let a=hC(r),o;o=n?new Rw(a):new Lw(a);let s=[t??PS(a)];return{dtype:i,shape:r,dataId:this.runWebGLProgram(o,[{shape:a,dtype:i,dataId:e}],i,s,!0,t).dataId}}runWebGLProgram(e,t,n,r,i=!1,a){let o=this.makeTensorInfo(e.outputShape,n),s=this.texData.get(o.dataId);if(e.packedOutput&&(s.isPacked=!0),e.outPackingScheme===kS.DENSE&&(s.texShape=(a??PS(e.outputShape)).map(e=>e*2)),e.outTexUsage!=null&&(s.usage=e.outTexUsage),S(o.shape)===0)return s.values=ie(o.dtype,0),o;let c=[],l=t.map(t=>{if(t.dtype===`complex64`)throw Error(`GPGPUProgram does not support complex64 input. For complex64 dtypes, please separate the program into real and imaginary parts.`);let n=this.texData.get(t.dataId);if(n.texture==null){if(!e.packedInputs&&S(t.shape)<=O().getNumber(`WEBGL_SIZE_UPLOAD_UNIFORM`))return{shape:t.shape,texData:null,isUniform:!0,uniformValues:n.values};e.packedInputs&&(n.isPacked=!0,n.shape=t.shape)}if(this.uploadToGPU(t.dataId),!!n.isPacked!=!!e.packedInputs)t=n.isPacked?this.unpackTensor(t):this.packTensor(t),c.push(t),n=this.texData.get(t.dataId);else if(n.isPacked&&!vC(n.shape,t.shape)){let e=t,r=t.shape;t.shape=n.shape,t=this.packedReshape(t,r),c.push(t),n=this.texData.get(t.dataId),e.shape=r}return{shape:t.shape,texData:n,isUniform:!1}});this.uploadToGPU(o.dataId);let u={shape:o.shape,texData:s,isUniform:!1},d=Fw(e,l,u),f=this.getAndSaveBinary(d,()=>jw(this.gpgpu,e,l,u)),p=this.activeTimers!=null,m;p&&(m=this.startTimer()),O().get(`ENGINE_COMPILE_ONLY`)||Pw(this.gpgpu,f,l,u,r),c.forEach(e=>this.disposeIntermediateTensorInfo(e)),p&&(m=this.endTimer(m),this.activeTimers.push({name:e.constructor.name,query:this.getQueryTime(m)}));let h=O().getNumber(`WEBGL_FLUSH_THRESHOLD`);if(h>0){let e=ni();e-this.lastGlFlushTime>h&&(this.gpgpu.gl.flush(),this.lastGlFlushTime=e)}if(!O().getBool(`WEBGL_LAZILY_UNPACK`)&&s.isPacked&&i===!1){let e=this.unpackTensor(o);return this.disposeIntermediateTensorInfo(o),e}return o}compileAndRun(e,t,n,r,i=!1){return n||=t[0].dtype,this.runWebGLProgram(e,t,n,r,i)}getAndSaveBinary(e,t){return e in this.binaryCache||(this.binaryCache[e]=t()),this.binaryCache[e]}getTextureManager(){return this.textureManager}dispose(){this.disposed||=(O().getBool(`IS_TEST`)||Object.keys(this.binaryCache).forEach(e=>{this.gpgpu.deleteProgram(this.binaryCache[e].webGLProgram),delete this.binaryCache[e]}),this.textureManager.dispose(),this.canvas!=null&&typeof HTMLCanvasElement<`u`&&this.canvas instanceof HTMLCanvasElement?this.canvas.remove():this.canvas=null,this.gpgpuCreatedLocally&&(this.gpgpu.program=null,this.gpgpu.dispose()),!0)}floatPrecision(){return this.floatPrecisionValue??=P(()=>{if(!O().get(`WEBGL_RENDER_FLOAT32_ENABLED`)){let e=O().getBool(`DEBUG`);O().set(`DEBUG`,!1);let t=this.abs(G(1e-8)).dataSync()[0];if(O().set(`DEBUG`,e),t>0)return 32}return 16}),this.floatPrecisionValue}epsilon(){return this.floatPrecision()===32?cO:lO}uploadToGPU(e){let t=this.texData.get(e),{shape:n,dtype:r,values:i,texture:a,usage:o,isPacked:s}=t;if(a!=null)return;let c=this.activeTimers!=null,l;c&&(l=ni());let u=t.texShape;if(u??(u=gC(n,s),t.texShape=u),i!=null){let e=hC(n),a,o=u[1],d=u[0],f=i instanceof Uint8Array||i instanceof Uint8ClampedArray;(s||!f)&&([o,d]=FS(u[0],u[1])),a=s?new Uw(e,f):new Hw(e,f);let p=f?[d,o]:u,m=this.makeTensorInfo(p,r),h=this.texData.get(m.dataId);f?h.usage=AS.PIXELS:h.usage=AS.UPLOAD,h.texShape=p,this.gpgpu.uploadDenseMatrixToTexture(this.getTexture(m.dataId),o,d,i);let g=[[d,o]],_=this.runWebGLProgram(a,[m],r,g,!0),v=this.texData.get(_.dataId);t.texShape=v.texShape,t.isPacked=v.isPacked,t.usage=v.usage,O().get(`ENGINE_COMPILE_ONLY`)?this.disposeData(_.dataId):(t.texture=v.texture,t.values=null,this.texData.delete(_.dataId)),this.disposeIntermediateTensorInfo(m),c&&(this.uploadWaitMs+=ni()-l)}else t.texture=this.acquireTexture(u,o,r,s)}convertAndCacheOnCPU(e,t){let n=this.texData.get(e),{dtype:r}=n;return t!=null&&(n.values=gO(t,r)),n.values}acquireTexture(e,t,n,r){if(this.numBytesInGPU+=this.computeBytes(e,n),!this.warnedAboutMemory&&this.numBytesInGPU>this.numMBBeforeWarning*1024*1024){let e=(this.numBytesInGPU/1024/1024).toFixed(2);this.warnedAboutMemory=!0,console.warn(`High memory usage in GPU: ${e} MB, most likely due to a memory leak`)}return this.textureManager.acquireTexture(e,t,r)}computeBytes(e,t){return e[0]*e[1]*le(t)}checkCompileCompletion(){for(let[,e]of Object.entries(this.binaryCache))this.checkCompletion_(e)}async checkCompileCompletionAsync(){let e=[];if(this.gpgpu.parallelCompilationExtension){for(let[,t]of Object.entries(this.binaryCache))e.push(this.checkCompletionAsync_(t));return Promise.all(e)}else{for(let[,t]of Object.entries(this.binaryCache)){let n=new Promise(e=>{try{this.checkCompletion_(t),e(!0)}catch(e){throw e}});e.push(n)}return Promise.all(e)}}async checkCompletionAsync_(e){return this.gpgpu.gl.getProgramParameter(e.webGLProgram,this.gpgpu.parallelCompilationExtension.COMPLETION_STATUS_KHR)?this.checkCompletion_(e):(await dv(),this.checkCompletionAsync_(e))}checkCompletion_(e){if(this.gpgpu.gl.getProgramParameter(e.webGLProgram,this.gpgpu.gl.LINK_STATUS)===!1)throw console.log(this.gpgpu.gl.getProgramInfoLog(e.webGLProgram)),this.gpgpu.gl.getShaderParameter(e.fragmentShader,this.gpgpu.gl.COMPILE_STATUS)===!1?(qS(e.source,this.gpgpu.gl.getShaderInfoLog(e.fragmentShader)),Error(`Failed to compile fragment shader.`)):Error(`Failed to link vertex and fragment shaders.`);return!0}getUniformLocations(){for(let e of Object.values(this.binaryCache)){this.gpgpu.buildVao(e.webGLProgram);let{variablesLocations:t,customUniformLocations:n,infLoc:r,nanLoc:i,outShapeLocation:a,outShapeStridesLocation:o,outTexShapeLocation:s}=Mw(this.gpgpu,e.program,e.webGLProgram);e.variablesLocations=t,e.customUniformLocations=n,e.infLoc=r,e.nanLoc=i,e.outShapeLocation=a,e.outShapeStridesLocation=o,e.outTexShapeLocation=s}}createTensorFromGPUData(e,t,n){e.channels=e.channels||`RGBA`;let{texture:r,height:i,width:a,channels:o}=e,s=fa().backend;if(!s.gpgpu.gl.isTexture(r))throw Error(`The texture is invalid. Also, please make sure the texture and the TFJS WebGL backend are using the same canvas. If you want to use your own custom canvas, you have to create and use the custom TFJS WebGL backend created from the canvas through 'new tf.MathBackendWebGL(customCanvas)'.`);let c=s.writeTexture(r,t,n,i,a,o);return fa().makeTensorFromDataId(c,t,n,s)}};hO.nextDataId=0;function gO(e,t){if(t===`float32`||t===`complex64`)return e;if(t===`int32`||t===`bool`){let n=t===`int32`?new Int32Array(e.length):new Uint8Array(e.length);for(let t=0;t<n.length;++t)n[t]=Math.round(e[t]);return n}else throw Error(`Unknown dtype ${t}`)}Qi()&&ha(`webgl`,()=>new hO,2);var _O=`
  if (isnan(a)) return a;
  if (isnan(b)) return b;
`,vO=class{constructor(e,t,n){this.variableNames=[`A`,`B`],this.outputShape=W(t,n),this.enableShapeUniforms=Iw(this.outputShape.length),this.userCode=`
      float binaryOperation(float a, float b) {
        ${e}
      }

      void main() {
        float a = getAAtOutCoords();
        float b = getBAtOutCoords();
        setOutput(binaryOperation(a, b));
      }
    `}},yO=`
  result.r = isNaN.r ? NAN : result.r;
  result.g = isNaN.g ? NAN : result.g;
  result.b = isNaN.b ? NAN : result.b;
  result.a = isNaN.a ? NAN : result.a;
`,bO=class{constructor(e,t,n,r=!1){this.variableNames=[`A`,`B`],this.supportsBroadcasting=!0,this.packedInputs=!0,this.packedOutput=!0,this.outputShape=W(t,n);let i=this.outputShape.length;this.enableShapeUniforms=Iw(i);let a=``;if(r)if(i===0||S(this.outputShape)===1)a=`
          result.y = 0.;
          result.z = 0.;
          result.w = 0.;
        `;else if(a=`
          ${Z(i)} coords = getOutputCoords();
        `,i===1)this.enableShapeUniforms?a+=`
            result.y = (coords + 1) >= outShape ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `:a+=`
            result.y = (coords + 1) >= ${this.outputShape[0]} ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `;else{let e=ND(`coords`,i);this.enableShapeUniforms?a+=`
            bool nextRowOutOfBounds =
              (${e[i-2]} + 1) >= outShape[${i} - 2];
            bool nextColOutOfBounds =
              (${e[i-1]} + 1) >= outShape[${i} - 1];
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `:a+=`
            bool nextRowOutOfBounds =
              (${e[i-2]} + 1) >= ${this.outputShape[i-2]};
            bool nextColOutOfBounds =
              (${e[i-1]} + 1) >= ${this.outputShape[i-1]};
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `}this.userCode=`
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${e}
      }

      void main() {
        vec4 a = getAAtOutCoords();
        vec4 b = getBAtOutCoords();

        vec4 result = binaryOperation(a, b);
        ${a}

        setOutput(result);
      }
    `}};function xO(e){let{inputs:t,backend:n}=e,{x:r}=t;return n.incRef(r.dataId),{dataId:r.dataId,shape:r.shape,dtype:r.dtype}}var SO={kernelName:Bt,backendName:`webgl`,kernelFunc:xO};function CO(e){let{inputs:t,backend:n}=e,{real:r,imag:i}=t,a=n.makeTensorInfo(r.shape,`complex64`),o=n.texData.get(a.dataId);return o.complexTensorInfos={real:xO({inputs:{x:r},backend:n}),imag:xO({inputs:{x:i},backend:n})},a}var wO={kernelName:at,backendName:`webgl`,kernelFunc:CO},TO=`return (a < 0.) ? b * a : a;`,EO=`
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;function DO(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{alpha:a}=r,o=n.makeTensorInfo([],`float32`,$r(a,`float32`)),s=O().getBool(`WEBGL_PACK_BINARY_OPERATIONS`)?new bO(EO,i.shape,o.shape):new vO(TO,i.shape,o.shape),c=n.runWebGLProgram(s,[i,o],`float32`);return n.disposeIntermediateTensorInfo(o),c}var OO={kernelName:Kt,backendName:`webgl`,kernelFunc:DO},kO=`return (a < 0.) ? b * a : a;`,AO=`
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;function jO(e){let{inputs:t,backend:n}=e,{x:r,alpha:i}=t,a=O().getBool(`WEBGL_PACK_BINARY_OPERATIONS`)?new bO(AO,r.shape,i.shape):new vO(kO,r.shape,i.shape);return n.runWebGLProgram(a,[r,i],`float32`)}var MO={kernelName:xn,backendName:`webgl`,kernelFunc:jO},NO=`if (isnan(x)) return x;`;function Q({opSnippet:e,packedOpSnippet:t,cpuKernelImpl:n,dtype:r}){return({inputs:i,backend:a})=>{let{x:o}=i,s=a,c=r||o.dtype;if(s.shouldExecuteOnCPU([o])&&n!=null){let e=n(s.texData.get(o.dataId).values,c);return s.makeTensorInfo(o.shape,c,e)}let l=O().getBool(`WEBGL_PACK_UNARY_OPERATIONS`)&&t!=null,u;return u=l?new aO(o.shape,t):new GD(o.shape,e),s.runWebGLProgram(u,[o],c)}}function PO({opSnippet:e,packedOpSnippet:t,checkOutOfBounds:n=!1,supportsComplex:r=!1,cpuKernelImpl:i,dtype:a}){return({inputs:o,backend:s})=>{let{a:c,b:l}=o,u=s;if(r&&c.dtype===`complex64`){let t=u.texData.get(c.dataId),n=u.texData.get(l.dataId),[r,i]=[[t.complexTensorInfos.real,n.complexTensorInfos.real],[t.complexTensorInfos.imag,n.complexTensorInfos.imag]].map(t=>{let[n,r]=t,i={dataId:n.dataId,dtype:n.dtype,shape:c.shape},a={dataId:r.dataId,dtype:r.dtype,shape:l.shape},o=new vO(e,c.shape,l.shape);return u.runWebGLProgram(o,[i,a],Fi(n.dtype,r.dtype))}),a=CO({inputs:{real:r,imag:i},backend:u});return u.disposeIntermediateTensorInfo(r),u.disposeIntermediateTensorInfo(i),a}let d=a||Fi(c.dtype,l.dtype);if((c.dtype===`string`||l.dtype===`string`||u.shouldExecuteOnCPU([c,l]))&&i!=null){let e=u.texData.get(c.dataId).values,t=u.texData.get(l.dataId).values,n=c.dtype===`string`?gy(e):e,r=c.dtype===`string`?gy(t):t,[a,o]=i(c.shape,l.shape,n,r,d),s=u.makeTensorInfo(o,d),f=u.texData.get(s.dataId);return f.values=a,s}let f=O().getBool(`WEBGL_PACK_BINARY_OPERATIONS`)&&t!=null,p;return p=f?new bO(t,c.shape,l.shape,n):new vO(e,c.shape,l.shape),u.runWebGLProgram(p,[c,l],d)}}function FO(e,t=!1){if(e===`linear`)return t?eO:qD;if(e===`relu`)return t?nO:XD;if(e===`elu`)return t?tO:YD;if(e===`relu6`)return t?rO:ZD;if(e===`prelu`)return t?AO:kO;if(e===`leakyrelu`)return t?EO:TO;if(e===`sigmoid`)return t?iO:$D;throw Error(`Activation ${e} has not been implemented for the WebGL backend.`)}var IO=class{constructor(e,t,n,r=!1,i=!1,a=!1,o=null,s=!1,c=!1){this.variableNames=[`matrixA`,`matrixB`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=n,this.enableShapeUniforms=Iw(this.outputShape.length);let l=r?e[1]:e[2],u=Math.ceil(l/2),d=r?`i * 2, rc.y`:`rc.y, i * 2`,f=i?`rc.z, i * 2`:`i * 2, rc.z`,p=r?[`a.xxyy`,`a.zzww`]:[`a.xxzz`,`a.yyww`],m=i?[`b.xzxz`,`b.ywyw`]:[`b.xyxy`,`b.zwzw`],h=``,g=``;o&&(h=s?`vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${o}
        }`:c?`vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${o}
        }`:`vec4 activation(vec4 x) {
          ${o}
        }`,g=`result = activation(result);`);let _=a?`result += getBiasAtOutCoords();`:``;a&&this.variableNames.push(`bias`),s&&this.variableNames.push(`preluActivationWeights`),c&&this.variableNames.push(`leakyreluAlpha`);let v=`rc.x`,y=`rc.x`;e[0]<t[0]?v=`imod(rc.x, ${e[0]})`:t[0]<e[0]&&(y=`imod(rc.x, ${t[0]})`),this.userCode=`
      ${h}
      // Don't use uniform for sharedDimensionPacked for performance.
      const float sharedDimension = ${u}.0;

      vec4 dot2x2ARowBCol(ivec3 rc) {
        vec4 result = vec4(0);
        int batchA = ${v};
        int batchB = ${y};
        for (int i = 0; i < ${u}; i++) {
          vec4 a = getMatrixA(batchA, ${d});
          vec4 b = getMatrixB(batchB, ${f});

          // These swizzled products need to be separately added.
          // See: https://github.com/tensorflow/tfjs/issues/1735
          result += (${p[0]} * ${m[0]});
          result += (${p[1]} * ${m[1]});
        }
        return result;
      }

      void main() {
        ivec3 rc = getOutputCoords();
        vec4 result = dot2x2ARowBCol(rc);

        ${_}

        ${g}

        setOutput(result);
      }
    `}},LO={REAL:`return areal * breal - aimag * bimag;`,IMAG:`return areal * bimag + aimag * breal;`},RO=class{constructor(e,t,n){this.variableNames=[`AReal`,`AImag`,`BReal`,`BImag`],this.outputShape=W(t,n),this.userCode=`
      float binaryOpComplex(
          float areal, float aimag, float breal, float bimag) {
        ${e}
      }

      void main() {
        float areal = getARealAtOutCoords();
        float aimag = getAImagAtOutCoords();
        float breal = getBRealAtOutCoords();
        float bimag = getBImagAtOutCoords();
        setOutput(binaryOpComplex(areal, aimag, breal, bimag));
      }
    `}},zO=`return a * b;`;function BO(e){let{inputs:t,backend:n}=e,{a:r,b:i}=t,a=Fi(r.dtype,i.dtype);if(r.dtype===`complex64`){let e=n.texData.get(r.dataId),t=n.texData.get(i.dataId),a=new RO(LO.REAL,r.shape,i.shape),o=new RO(LO.IMAG,r.shape,i.shape),s=[{dataId:e.complexTensorInfos.real.dataId,dtype:e.complexTensorInfos.real.dtype,shape:r.shape},{dataId:e.complexTensorInfos.imag.dataId,dtype:e.complexTensorInfos.imag.dtype,shape:r.shape},{dataId:t.complexTensorInfos.real.dataId,dtype:t.complexTensorInfos.real.dtype,shape:i.shape},{dataId:t.complexTensorInfos.imag.dataId,dtype:t.complexTensorInfos.imag.dtype,shape:i.shape}],c=n.runWebGLProgram(a,s,`float32`),l=n.runWebGLProgram(o,s,`float32`),u=CO({inputs:{real:c,imag:l},backend:n});return n.disposeIntermediateTensorInfo(c),n.disposeIntermediateTensorInfo(l),u}if(n.shouldExecuteOnCPU([r,i])){let e=n.texData.get(r.dataId),t=n.texData.get(i.dataId),[o,s]=aD(r.shape,i.shape,e.values,t.values,a),c=n.makeTensorInfo(s,a),l=n.texData.get(c.dataId);return l.values=o,c}let o;return o=O().getBool(`WEBGL_PACK_BINARY_OPERATIONS`)?new bO(zO,r.shape,i.shape):new vO(zO,r.shape,i.shape),n.runWebGLProgram(o,[r,i],a)}var VO={kernelName:fn,backendName:`webgl`,kernelFunc:BO};function HO(e,t,n){let r=[pC(e.shape),...mC(e.shape)],i={dtype:e.dtype,shape:r,dataId:e.dataId},a=new ID([pC(t),...mC(t)],r),o=[r],s=n.runWebGLProgram(a,[i],e.dtype,o,!0);return{dataId:s.dataId,shape:t,dtype:s.dtype}}function $(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{shape:a}=r,o=n,s=S(i.shape),c=ne(a,s),l=S(c);y(s===l,()=>`The new shape (${c}) has ${l} elements and the old shape (${i.shape}) has ${s} elements. The new shape and old shape must have the same number of elements.`);let u=o.texData.get(i.dataId);return u.isPacked&&!vC(i.shape,c)&&!(u.texture!==null&&vC(u.shape,c))?HO(i,c,o):(o.incRef(i.dataId),{dataId:i.dataId,shape:c,dtype:i.dtype})}var UO={kernelName:An,backendName:`webgl`,kernelFunc:$},WO=class{constructor(e,t){this.variableNames=[`x`];let{windowSize:n,batchSize:r,inSize:i,outSize:a}=e;this.outputShape=[r,a];let o=Math.floor(n/4)*4,s=n%4,c=`sumValue += dot(values, ones);`;if(t!=null){let e=1/t;c=`sumValue += dot(values * ${ee(e)?e.toPrecision(2):e}, ones);`}let l=``;i%n>0&&(l=`
        if (inIdx < 0 || inIdx >= ${i}) {
          return 0.0;
        }
      `),this.userCode=`
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${l}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${n};

        float sumValue = 0.0;

        for (int i = 0; i < ${o}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${c}
        }

        int inIdx = inOffset + ${o};
        if (${s===1}) {
          vec4 values = vec4(getValue(batch, inIdx), 0.0, 0.0, 0.0);

          ${c}
        } else if (${s===2}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1), 0.0, 0.0);

          ${c}
        } else if (${s===3}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2), 0.0);

          ${c}
        }
        setOutput(sumValue);
      }
    `}},GO=class{constructor(e,t){this.variableNames=[`x`];let{windowSize:n,batchSize:r,inSize:i,outSize:a}=e;this.outputShape=[r,a];let o=`0.0`,s=``;t===`prod`?o=`1.0`:t===`min`?(o=`1.0 / 1e-20`,s=`min`):t===`max`&&(o=`-1.0 / 1e-20`,s=`max`);let c=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;t===`sum`?c=`sumValue`:t===`prod`?c=`prodValue`:t===`all`?c=`allValue`:t===`any`&&(c=`anyValue`);let l=Math.floor(n/4)*4,u=n%4,d=`
      if (${t===`sum`}) {
        sumValue += dot(values, ones);
      } else if (${t===`prod`}) {
        vec2 tmp = vec2(values[0], values[1]) * vec2(values[2], values[3]);
        prodValue *= tmp[0] * tmp[1];
      } else {
        minMaxValue = ${s}(values, minMaxValue);
        if (${t===`min`} || ${t===`max`}) {
          minMaxValue = ${s}(values, minMaxValue);
          bvec4 isNaN = isnan(values);
          if (isNaN.r || isNaN.g || isNaN.b || isNaN.a) {
            minMaxValue = vec4(NAN);
          }
        }
      }
    `,f=`vec4`;t===`all`?(o=`1.0`,d=`
        bool reducedAllValue = all(values);
        float floatedReducedAllValue = float(reducedAllValue);
        allValue = float(allValue >= 1.0 && floatedReducedAllValue >= 1.0);
      `,f=`bvec4`):t===`any`&&(o=`0.0`,d=`
        bool reducedAnyValue = any(values);
        float floatedReducedAnyValue = float(reducedAnyValue);
        anyValue = float(anyValue >= 1.0 || floatedReducedAnyValue >= 1.0);
      `,f=`bvec4`);let p=``;i%n>0&&(p=`
        if (inIdx < 0 || inIdx >= ${i}) {
          return initializationValue;
        }
      `),this.userCode=`
      const float initializationValue = ${o};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${p}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${n};

        vec4 minMaxValue = vec4(${o});
        float prodValue = 1.0;
        float sumValue = 0.0;
        float allValue = 1.0;
        float anyValue = 0.0;

        for (int i = 0; i < ${l}; i += 4) {
          int inIdx = inOffset + i;
          ${f} values = ${f}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${d}
        }

        int inIdx = inOffset + ${l};
        if (${u===1}) {
          ${f} values = ${f}(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          ${d}
        } else if (${u===2}) {
          ${f} values = ${f}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          ${d}
        } else if (${u===3}) {
          ${f} values = ${f}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          ${d}
        }
        setOutput(${c});
      }
    `}};function KO(e){let t=[];for(;t.length===0||t[t.length-1].outSize!==1;){let n=t.length?t[t.length-1].outSize:e[1],r=yv(n);t.push({inSize:n,windowSize:r,outSize:Math.ceil(n/r)})}return t}function qO(e,t,n,r){let i=KO(e.shape),a=e;for(let o=0;o<i.length;o++){let{inSize:s,windowSize:c,outSize:l}=i[o],u,d;u=n===`mean`?o===0?new WO({windowSize:c,inSize:s,batchSize:e.shape[0],outSize:l},s):new WO({windowSize:c,inSize:s,batchSize:e.shape[0],outSize:l}):new GO({windowSize:c,inSize:s,batchSize:e.shape[0],outSize:l},n),d=a,a=r.runWebGLProgram(u,[a],t),d.dataId!==e.dataId&&r.disposeIntermediateTensorInfo(d)}return a}var JO=class{constructor(e,t){this.variableNames=[`A`];let n=Array(e.length);for(let r=0;r<n.length;r++)n[r]=e[t[r]];this.outputShape=n,this.rank=n.length;let r=Z(this.rank),i=YO(t);this.userCode=`
    void main() {
      ${r} resRC = getOutputCoords();
      setOutput(getA(${i}));
    }
    `}};function YO(e){let t=e.length;if(t>6)throw Error(`Transpose for rank ${t} is not yet supported`);let n=[`resRC.x`,`resRC.y`,`resRC.z`,`resRC.w`,`resRC.u`,`resRC.v`],r=Array(t);for(let t=0;t<e.length;t++)r[e[t]]=n[t];return r.join()}var XO=class{constructor(e,t){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0;let n=Array(e.length);for(let r=0;r<n.length;r++)n[r]=e[t[r]];if(this.outputShape=n,this.rank=n.length,this.rank>6)throw Error(`Packed transpose for rank ${this.rank} is not yet supported.`);let r=Z(this.rank),i=MD(`rc`,this.rank),a=Array(this.rank);for(let e=0;e<t.length;e++)a[t[e]]=i[e];let o=`vec2(${a.slice(-2).join()})`,s=`++${i[this.rank-1]} < ${n[this.rank-1]}`,c=`getChannel(getA(${a.join()}), ${o})`;this.userCode=`
    void main() {
      ${r} rc = getOutputCoords();
      vec4 result = vec4(0.);
      result[0] = ${c};
      if(${s}) {
        result[1] = ${c};
      }
      --${i[this.rank-1]};
      if(++${i[this.rank-2]} < ${n[this.rank-2]}) {
        result[2] = ${c};
        if(${s}) {
          result[3] = ${c};
        }
      }
      setOutput(result);
    }
    `}};function ZO(e,t,n){let r=O().getBool(`WEBGL_PACK_ARRAY_OPERATIONS`)?new XO(e.shape,t):new JO(e.shape,t);return n.runWebGLProgram(r,[e],e.dtype)}function QO(e,t,n,r){let i=t,a=e.shape.length,o=D(i,e.shape),s=o,c=Ol(s,a),l=c!=null,u=e;l&&(u=ZO(e,c,r),s=Al(s.length,a)),Dl(`sum`,s,a);let[d,f]=Tl(u.shape,s),p=d;n&&(p=El(d,o));let m=S(f),h=S(e.shape)/m,g=$({inputs:{x:u},attrs:{shape:[h,m]},backend:r}),_=qO(g,Ii(e.dtype),`sum`,r),v=$({inputs:{x:_},attrs:{shape:p},backend:r});return r.disposeIntermediateTensorInfo(g),r.disposeIntermediateTensorInfo(_),l&&r.disposeIntermediateTensorInfo(u),v}function $O(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;return QO(i,a,o,n)}var ek={kernelName:`Sum`,backendName:`webgl`,kernelFunc:$O};function tk(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{perm:a}=r,o=n,s=i.shape.length,c=Array(s);for(let e=0;e<c.length;e++)c[e]=i.shape[a[e]];let l;if(o.shouldExecuteOnCPU([i])){let e=o.texData.get(i.dataId).values,t=AD(e,i.shape,i.dtype,a,c);l=o.makeTensorInfo(c,i.dtype);let n=o.texData.get(l.dataId);n.values=t}else l=ZO(i,a,o);return l}var nk={kernelName:hr,backendName:`webgl`,kernelFunc:tk};function rk({a:e,b:t,transposeA:n,transposeB:r,backend:i,bias:a=null,preluActivationWeights:o=null,leakyreluAlpha:s=0,activation:c=null}){let l=e.shape.length,u=t.shape.length,d=n?e.shape[l-2]:e.shape[l-1],f=r?t.shape[u-1]:t.shape[u-2],p=n?e.shape[l-1]:e.shape[l-2],m=r?t.shape[u-2]:t.shape[u-1],h=e.shape.slice(0,-2),g=t.shape.slice(0,-2),_=S(h),v=S(g),b=W(e.shape.slice(0,-2),t.shape.slice(0,-2)).concat([p,m]);y(d===f,()=>`Error in matMul: inner shapes (${d}) and (${f}) of Tensors with shapes ${e.shape} and ${t.shape} and transposeA=${n} and transposeB=${r} must match.`);let x=n?[_,d,p]:[_,p,d],C=r?[v,m,f]:[v,f,m],w=$({inputs:{x:e},backend:i,attrs:{shape:x}}),ee=$({inputs:{x:t},backend:i,attrs:{shape:C}}),T=[w,ee],E=Math.max(_,v),te=n?w.shape[1]:w.shape[2],ne=a!=null,D=o!=null,re=c===`leakyrelu`,ie=c==null?null:FO(c,!0),ae=ne||D||re||ie!=null,oe;if((p===1||m===1)&&te>1e3&&ae===!1){let e=w,t=ee;n&&(e=tk({inputs:{x:w},backend:i,attrs:{perm:[0,2,1]}}),T.push(e)),r&&(t=tk({inputs:{x:ee},backend:i,attrs:{perm:[0,2,1]}}),T.push(t));let a=m!==1,o=m===1,s=e;a&&(s=$({inputs:{x:e},backend:i,attrs:{shape:[E,te,1]}}),T.push(s));let c=m===1?2:1,l=t;o&&(l=$({inputs:{x:t},backend:i,attrs:{shape:[E,1,te]}}),T.push(l));let u=BO({inputs:{a:s,b:l},backend:i});oe=$O({inputs:{x:u},backend:i,attrs:{axis:c,keepDims:!0}}),T.push(u)}else{let c=Fi(e.dtype,t.dtype),l=new IO(x,C,[E,p,m],n,r,ne,ie,D,re),u=[w,ee];if(a!=null&&u.push(a),D&&u.push(o),re){let e=i.makeTensorInfo([],`float32`,$r(s,`float32`));u.push(e),T.push(e)}oe=i.runWebGLProgram(l,u,c)}let se=$({inputs:{x:oe},backend:i,attrs:{shape:b}});T.push(oe);for(let e of T)i.disposeIntermediateTensorInfo(e);return se}function ik(e){let{inputs:t,backend:n,attrs:r}=e,{a:i,b:a,bias:o,preluActivationWeights:s}=t,{transposeA:c,transposeB:l,activation:u,leakyreluAlpha:d}=r;return rk({a:i,b:a,transposeA:c,transposeB:l,backend:n,bias:o,preluActivationWeights:s,leakyreluAlpha:d,activation:u})}var ak={kernelName:Cr,backendName:`webgl`,kernelFunc:ik},ok=`return abs(x);`;function sk(e){let{inputs:t,backend:n}=e,{x:r}=t;if(n.shouldExecuteOnCPU([r])&&r.dtype!==`complex64`){let e=gD(n.texData.get(r.dataId).values);return n.makeTensorInfo(r.shape,r.dtype,e)}let i;return i=O().getBool(`WEBGL_PACK_UNARY_OPERATIONS`)?new aO(r.shape,ok):new GD(r.shape,ok),n.runWebGLProgram(i,[r],r.dtype)}var ck={kernelName:`Abs`,backendName:`webgl`,kernelFunc:sk},lk={kernelName:Le,backendName:`webgl`,kernelFunc:Q({opSnippet:KD+`
  if (abs(x) > 1.) {
    return NAN;
  }
  return acos(x);
`})},uk={kernelName:Re,backendName:`webgl`,kernelFunc:Q({opSnippet:KD+`
  if (x < 1.0) return NAN;
return log(x + sqrt(x * x - 1.0));`})},dk=`return a + b;`,fk={kernelName:`Add`,backendName:`webgl`,kernelFunc:PO({opSnippet:dk,packedOpSnippet:dk,supportsComplex:!0,cpuKernelImpl:LE})},pk=class{constructor(e,t){this.outputShape=[],this.outputShape=e,this.variableNames=t.map((e,t)=>`T${t}`);let n=[];this.variableNames.forEach(e=>{n.push(`float v${e} = get${e}AtOutCoords();`)});let r=this.variableNames.map(e=>`v${e}`).join(` + `);this.userCode=`
      void main() {
        ${n.join(`
        `)}

        float result = ${r};
        setOutput(result);
      }
    `}},mk=class{constructor(e,t){this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.variableNames=t.map((e,t)=>`T${t}`);let n=[];this.variableNames.forEach(e=>{n.push(`vec4 v${e} = get${e}AtOutCoords();`)});let r=this.variableNames.map(e=>`v${e}`).join(` + `);this.userCode=`
      void main() {
        ${n.join(`
        `)}

        vec4 result = ${r};
        setOutput(result);
      }
    `}};function hk(e){let{inputs:t,backend:n}=e,r=t;if(r.length===1)return xO({inputs:{x:r[0]},backend:n});if(r.length>O().getNumber(`WEBGL_MAX_TEXTURES_IN_SHADER`)){let e=Math.floor(r.length/2);return hk({inputs:[hk({inputs:r.slice(0,e),backend:n}),hk({inputs:r.slice(e),backend:n})],backend:n})}let i=r.map(e=>e.dtype).reduce((e,t)=>Fi(e,t)),a=r.map(e=>e.shape),o=O().getBool(`WEBGL_PACK`)?new mk(r[0].shape,a):new pk(r[0].shape,a);return n.runWebGLProgram(o,r,i)}var gk={kernelName:ze,backendName:`webgl`,kernelFunc:hk};function _k(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r,s=i.shape.length,c=D(a,i.shape),l=c,u=Ol(l,s),d=i;u!=null&&(d=tk({inputs:{x:i},backend:n,attrs:{perm:u}}),l=Al(l.length,s)),Dl(`all`,l,s);let[f,p]=Tl(d.shape,l),m=S(p),h=$({inputs:{x:d},backend:n,attrs:{shape:[-1,m]}}),g=qO(h,h.dtype,`all`,n),_;if(o){let e=El(f,c);_=$({inputs:{x:g},backend:n,attrs:{shape:e}})}else _=$({inputs:{x:g},backend:n,attrs:{shape:f}});return n.disposeIntermediateTensorInfo(h),n.disposeIntermediateTensorInfo(g),u!=null&&n.disposeIntermediateTensorInfo(d),_}var vk={kernelName:`All`,backendName:`webgl`,kernelFunc:_k};function yk(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r,s=i.shape.length,c=D(a,i.shape),l=c,u=Ol(l,s),d=i;u!=null&&(d=tk({inputs:{x:i},backend:n,attrs:{perm:u}}),l=Al(l.length,s)),Dl(`any`,l,s);let[f,p]=Tl(d.shape,l),m=S(p),h=$({inputs:{x:d},backend:n,attrs:{shape:[-1,m]}}),g=qO(h,h.dtype,`any`,n),_;if(o){let e=El(f,c);_=$({inputs:{x:g},backend:n,attrs:{shape:e}})}else _=$({inputs:{x:g},backend:n,attrs:{shape:f}});return n.disposeIntermediateTensorInfo(h),n.disposeIntermediateTensorInfo(g),u!=null&&n.disposeIntermediateTensorInfo(d),_}var bk={kernelName:`Any`,backendName:`webgl`,kernelFunc:yk},xk=class{constructor(e,t,n){this.variableNames=[`A`];let{windowSize:r,batchSize:i,outSize:a}=e;n||this.variableNames.push(`bestIndicesA`),this.outputShape=[i,a];let o=t===`max`?`>`:`<`,s=n?`inOffset + i;`:`round(getBestIndicesA(batch, inOffset + i));`;this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${r};

        int bestIndex = inOffset;
        float bestValue = getA(batch, bestIndex);

        for (int i = 0; i < ${r}; i++) {
          int inIdx = ${s};
          float candidate = getA(batch, inIdx);
          if (candidate ${o} bestValue) {
            bestValue = candidate;
            bestIndex = inIdx;
          }
        }
        setOutput(float(bestIndex));
      }
    `}},Sk=class{constructor(e,t,n,r){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,y(e.length>2,()=>`Packed arg${n.charAt(0).toUpperCase()+n.slice(1)} supports only inputs with rank above 2.`);let i=e[e.length-1],a=Math.ceil(i/t);this.outputShape=e.slice(0,-1),a>1&&this.outputShape.push(a),r||this.variableNames.push(`bestIndicesA`);let o=this.outputShape,s=o.length,c=Z(s),l=ND(`coords`,s),u,d;if(a===1){d=s+1;let e=Z(d);u=`
        ${e} sourceLocR = ${e}(${l.join()}, 0);
        ++${l[s-1]};
        ${e} sourceLocG = ${e}(${l.join()}, 0);
        ++${l[s-2]};
        ${e} sourceLocA = ${e}(${l.join()}, 0);
        --${l[s-1]};
        ${e} sourceLocB = ${e}(${l.join()}, 0);
        --${l[s-2]};`}else d=s,u=`
        ${c} sourceLocR = coords;
        ++${l[s-1]};
        ${c} sourceLocG = coords;
        ++${l[s-2]};
        ${c} sourceLocA = coords;
        --${l[s-1]};
        ${c} sourceLocB = coords;
        --${l[s-2]};`;let f=[`x`,`y`,`z`,`w`,`u`,`v`].slice(0,d),p=`.`+f[d-1],m=f.map(e=>`int `+e),h=ND(`sourceLocR`,d-1).concat(`inIdx.r`),g=ND(`sourceLocG`,d-1).concat(`inIdx.g`),_=ND(`sourceLocB`,d-1).concat(`inIdx.b`),v=ND(`sourceLocA`,d-1).concat(`inIdx.a`),b=n===`max`?`greaterThan`:`lessThan`,x=r?``:`
          inIdx = round(vec4(getBestIndicesAChannel(${h.join()}),
                             getBestIndicesAChannel(${g.join()}),
                             getBestIndicesAChannel(${_.join()}),
                             getBestIndicesAChannel(${v.join()})));`,S=`vec4(
            getAChannel(${h.join()}),
            hasNextCol ? getAChannel(${g.join()}) : 0.,
            hasNextRow ? getAChannel(${_.join()}) : 0.,
            hasNextRow && hasNextCol ? getAChannel(${v.join()}) : 0.)`,C=r?``:`
      float getBestIndicesAChannel(${m.join()}) {
        return getChannel(getBestIndicesA(${f.join()}),
                                          vec2(${f.slice(-2).join()}));
      }`;this.userCode=`
      float getAChannel(${m.join()}) {
        return getChannel(getA(${f.join()}),
                               vec2(${f.slice(-2).join()}));
      }
      ${C}
      void main() {
        ${c} coords = getOutputCoords();
        bool hasNextCol = ${l[s-1]} < ${o[s-1]-1};
        bool hasNextRow = ${l[s-2]} < ${o[s-2]-1};
        ${u}
        ivec4 srcIdx = ivec4(sourceLocR${p}, sourceLocG${p},
          sourceLocB${p}, sourceLocA${p}) * ${t};
        ivec4 inIdx = srcIdx;
        vec4 bestIndex = vec4(inIdx);
        vec4 bestValue = ${S};

        for (int i = 0; i < ${t}; i++) {
          inIdx = srcIdx;
          ${x}
          vec4 candidate = ${S};
          bvec4 nan = isnan(candidate);
          bvec4 replace = bvec4(
            vec4(${b}(candidate, bestValue)) * (vec4(1.0) - vec4(nan)));

          bestValue = vec4(replace.x  ? candidate.x : bestValue.x,
                           replace.y  ? candidate.y : bestValue.y,
                           replace.z  ? candidate.z : bestValue.z,
                           replace.w  ? candidate.w : bestValue.w);
          bestIndex = mix(bestIndex, vec4(inIdx), vec4(replace));
          srcIdx++;
        }
        setOutput(bestIndex);
      }
    `}};function Ck(e,t,n,r=null){let i=t.shape[0],a=t.shape[1];r!=null&&(i=r.shape[0],a=r.shape[1]);let o=yv(a),s=new xk({windowSize:o,inSize:a,batchSize:i,outSize:Math.ceil(a/o)},n,r==null),c=[t];r!=null&&c.push(r);let l=e.runWebGLProgram(s,c,`int32`);if(l.shape[1]===1)return l;let u=Ck(e,t,n,l);return e.disposeIntermediateTensorInfo(l),u}function wk(e,t,n,r=null){let i=r==null?t.shape:r.shape,a=i[i.length-1],o=new Sk(i,yv(a),n,r==null),s=r==null?[t]:[t,r],c=e.runWebGLProgram(o,s,`int32`);if(c.shape.length===t.shape.length){let r=wk(e,t,n,c);return e.disposeIntermediateTensorInfo(c),r}return c}function Tk(e,t,n,r){let i=[n];if(Dl(`arg`+r.charAt(0).toUpperCase()+r.slice(1),i,t.shape.length),!O().getBool(`WEBGL_PACK_REDUCE`)||t.shape.length<=2){let n=[],a=e.texData.get(t.dataId),o=a!==null&&a.isPacked,s=t;o&&(s=e.unpackTensor(t),n.push(s));let[c,l]=Tl(s.shape,i),u=S(l),d=$({inputs:{x:s},backend:e,attrs:{shape:[-1,u]}});n.push(d);let f=Ck(e,d,r);n.push(f);let p=$({inputs:{x:f},backend:e,attrs:{shape:c}});return n.forEach(t=>e.disposeIntermediateTensorInfo(t)),p}return wk(e,t,r)}function Ek(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a}=r,o=D(a,i.shape),s=Ol(o,i.shape.length),c=i,l=[];s!=null&&(c=tk({inputs:{x:i},backend:n,attrs:{perm:s}}),l.push(c),o=Al(o.length,c.shape.length)),Dl(`argMax`,[o[0]],c.shape.length);let u=Tk(n,c,o[0],`max`);return l.forEach(e=>n.disposeIntermediateTensorInfo(e)),u}var Dk={kernelName:Be,backendName:`webgl`,kernelFunc:Ek};function Ok(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a}=r,o=D(a,i.shape),s=Ol(o,i.shape.length),c=i,l=[];s!=null&&(c=tk({inputs:{x:i},backend:n,attrs:{perm:s}}),l.push(c),o=Al(o.length,c.shape.length)),Dl(`argMin`,[o[0]],c.shape.length);let u=Tk(n,c,o[0],`min`);return l.forEach(e=>n.disposeIntermediateTensorInfo(e)),u}var kk={kernelName:Ve,backendName:`webgl`,kernelFunc:Ok},Ak={kernelName:He,backendName:`webgl`,kernelFunc:Q({opSnippet:KD+`
  if (abs(x) > 1.) {
    return NAN;
  }
  return asin(x);
`})},jk={kernelName:Ue,backendName:`webgl`,kernelFunc:Q({opSnippet:KD+`return log(x + sqrt(x * x + 1.0));`})},Mk={kernelName:We,backendName:`webgl`,kernelFunc:Q({opSnippet:KD+`
  return atan(x);
`})},Nk={kernelName:Ke,backendName:`webgl`,kernelFunc:PO({opSnippet:_O+`
  return atan(a, b);
`,packedOpSnippet:`
  vec4 result = atan(a, b);
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+yO+`
  return result;
`})},Pk={kernelName:Ge,backendName:`webgl`,kernelFunc:Q({opSnippet:KD+`
  if ((x < -1.0) || (x > 1.0)) return NAN;
return (log(1.0 + x) - log(1.0 - x)) / 2.0;`})},Fk=class{constructor(e,t,n,r=!1,i=!1){if(this.variableNames=[`x`],t===`avg`&&n)throw Error(`Cannot compute positions for average pool.`);let a=e.filterWidth,o=e.strideHeight,s=e.strideWidth,c=e.dilationHeight,l=e.dilationWidth,u=e.effectiveFilterHeight,d=e.effectiveFilterWidth,f=e.padInfo.top,p=e.padInfo.left;this.outputShape=e.outShape;let m=t===`avg`,h=`((batch  * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + d`,g=`(xR * ${e.inWidth} + xC) * ${e.inChannels} + d`,_=`0.0`;if(m||(_=`-1.0 / 1e-20`),n){this.userCode=`
        const ivec2 strides = ivec2(${o}, ${s});
        const ivec2 pads = ivec2(${f}, ${p});

        void main() {
          ivec4 coords = getOutputCoords();
          int batch = coords[0];
          int d = coords[3];

          ivec2 xRCCorner = coords.yz * strides - pads;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          // max/min x(?, ?, d) to get y(yR, yC, d).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;
          float avgValue = 0.0;

          for (int wR = 0; wR < ${u};
              wR += ${c}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${d};
                wC += ${l}) {
              int xC = xCCorner + wC;

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              float value = getX(batch, xR, xC, d);

              // If a min / max value has already been found, use it. If not,
              // use the current value.
              float currMinMaxValue = mix(
                  value, minMaxValue, minMaxValueFound);
              if (value >= currMinMaxValue) {
                minMaxValue = value;
                minMaxValueFound = 1.0;
                minMaxPosition = ${r?i?h:g:`wR * ${d} + wC`};
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;return}let v=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;t===`avg`&&(v=`avgValue / max(count, 1.0)`);let y=Math.floor(a/4)*4,b=a%4,x=`
      if (${m}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = max(values, minMaxValue);
      }
    `;this.userCode=`
      const ivec2 strides = ivec2(${o}, ${s});
      const ivec2 pads = ivec2(${f}, ${p});
      const float initializationValue = ${_};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xR, int xC, int d) {
        if (xC < 0 || xC >= ${e.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xR, xC, d);
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d = coords[3];

        ivec2 xRCCorner = coords.yz * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // max/min x(?, ?, d) to get y(yR, yC, d).
        // ? = to be determined
        vec4 minMaxValue = vec4(${_});
        float avgValue = 0.0;
        count = 0.0;

        for (int wR = 0; wR < ${u};
            wR += ${c}) {
          int xR = xRCorner + wR;

          if (xR < 0 || xR >= ${e.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${y}; wC += 4) {
            int xC = xCCorner + wC * ${l};

            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${l}, d),
              getValue(batch, xR, xC + 2 * ${l}, d),
              getValue(batch, xR, xC + 3 * ${l}, d)
            );

            ${x}
          }

          int xC = xCCorner + ${y};
          if (${b===1}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              initializationValue,
              initializationValue,
              initializationValue
            );

            ${x}
          } else if (${b===2}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${l}, d),
              initializationValue,
              initializationValue
            );

            ${x}
          } else if (${b===3}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${l}, d),
              getValue(batch, xR, xC + 2 * ${l}, d),
              initializationValue
            );

            ${x}
          }
        }
        setOutput(${v});
      }
    `}},Ik=class{constructor(e,t,n,r=!1,i=!1){if(this.variableNames=[`x`],t===`avg`&&n)throw Error(`Cannot compute positions for average pool.`);let a=e.filterWidth,o=e.strideDepth,s=e.strideHeight,c=e.strideWidth,l=e.dilationDepth,u=e.dilationHeight,d=e.dilationWidth,f=e.effectiveFilterDepth,p=e.effectiveFilterHeight,m=e.effectiveFilterWidth,h=e.padInfo.front,g=e.padInfo.top,_=e.padInfo.left;this.outputShape=e.outShape;let v=t===`avg`,y=`0.0`;if(v||(y=`-1.0 / 1e-20`),n){this.userCode=`
        const ivec3 strides =
            ivec3(${o}, ${s}, ${c});
        const ivec3 pads = ivec3(${h}, ${g}, ${_});

        void main() {
          ivec5 coords = getOutputCoords();
          int batch = coords.x;
          int ch = coords.u;

          ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
          int xDCorner = xCorner.x;
          int xRCorner = xCorner.y;
          int xCCorner = xCorner.z;

          // max/min x(?, ?, ?, ch) to get y(yD, yR, yC, ch).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;

          for (int wD = 0; wD < ${f};
              wD += ${l}) {
            int xD = xDCorner + wD;

            if (xD < 0 || xD >= ${e.inDepth}) {
              continue;
            }

            for (int wR = 0; wR < ${p};
                wR += ${u}) {
              int xR = xRCorner + wR;

              if (xR < 0 || xR >= ${e.inHeight}) {
                continue;
              }

              for (int wC = 0; wC < ${m};
                  wC += ${d}) {
                int xC = xCCorner + wC;

                if (xC < 0 || xC >= ${e.inWidth}) {
                  continue;
                }

                float value = getX(batch, xD, xR, xC, ch);

                // If a min / max value has already been found, use it. If not,
                // use the current value.
                float currMinMaxValue = mix(
                    value, minMaxValue, minMaxValueFound);
                if (value >= currMinMaxValue) {
                  minMaxValue = value;
                  minMaxValueFound = 1.0;
                  minMaxPosition = ${r?i?`(((batch * ${e.inDepth} + xD) * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + ch`:`((xD * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + ch`:`wD * ${p} * ${m} +
                      wR * ${m} + wC`};
                }
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;return}let b=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;t===`avg`&&(b=`avgValue / max(count, 1.0)`);let x=Math.floor(a/4)*4,S=a%4,C=`
      if (${v}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = max(values, minMaxValue);
      }
    `;this.userCode=`
      const ivec3 strides =
        ivec3(${o}, ${s}, ${c});
      const ivec3 pads = ivec3(${h}, ${g}, ${_});
      const float initializationValue = ${y};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xD, int xR, int xC, int ch) {
        if (xC < 0 || xC >= ${e.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xD, xR, xC, ch);
      }

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xDCorner = xCorner.x;
        int xRCorner = xCorner.y;
        int xCCorner = xCorner.z;

        // max/min x(?, ?, ?, d) to get y(yD, yR, yC, ch).
        // ? = to be determined
        vec4 minMaxValue = vec4(${y});
        float avgValue = 0.0;
        count = 0.0;

        for (int wD = 0; wD < ${f};
            wD += ${l}) {
          int xD = xDCorner + wD;

          if (xD < 0 || xD >= ${e.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${p};
            wR += ${u}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${x}; wC += 4) {
              int xC = xCCorner + wC * ${d};

              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${d}, ch),
                getValue(batch, xD, xR, xC + 2 * ${d}, ch),
                getValue(batch, xD, xR, xC + 3 * ${d}, ch)
              );

              ${C}
            }

            int xC = xCCorner + ${x};
            if (${S===1}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                initializationValue,
                initializationValue,
                initializationValue
              );

              ${C}
            } else if (${S===2}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${d}, ch),
                initializationValue,
                initializationValue
              );

              ${C}
            } else if (${S===3}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${d}, ch),
                getValue(batch, xD, xR, xC + 2 * ${d}, ch),
                initializationValue
              );

              ${C}
            }
          }
        }
        setOutput(${b});
      }
    `}};function Lk(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t;jC(i,`avgPool`);let{filterSize:a,strides:o,pad:s,dimRoundingMode:c}=r;y(Os(o,1),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${o} and dilations '1'`);let l=ms(i.shape,a,o,1,s,c);if(l.filterWidth===1&&l.filterHeight===1&&w(l.inShape,l.outShape))return xO({inputs:{x:i},backend:n});let u=new Fk(l,`avg`,!1);return n.runWebGLProgram(u,[i],`float32`)}var Rk={kernelName:qe,backendName:`webgl`,kernelFunc:Lk};function zk(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dimRoundingMode:c,dataFormat:l}=r,u=new Ik(hs(i.shape,a,o,[1,1,1],s,c,l),`avg`,!1);return n.runWebGLProgram(u,[i],`float32`)}var Bk={kernelName:Ye,backendName:`webgl`,kernelFunc:zk},Vk=class{constructor(e){this.variableNames=[`dy`],this.outputShape=e.inShape;let t=e.filterHeight,n=e.filterWidth,r=e.strideHeight,i=e.strideWidth,a=e.dilationHeight,o=e.dilationWidth,s=e.effectiveFilterHeight,c=e.effectiveFilterWidth,l=s-1-e.padInfo.top,u=c-1-e.padInfo.left,d=1/(t*n);this.userCode=`
      const ivec2 pads = ivec2(${l}, ${u});
      const float avgMultiplier = float(${d});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${s};
            wR += ${a}) {
          float dyR = float(dyRCorner + wR) / ${r}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${c};
            wC+= ${o}) {
            float dyC = float(dyCCorner + wC) / ${i}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);

            dotProd += dyValue * avgMultiplier;
          }
        }
        setOutput(dotProd);
      }
    `}},Hk=class{constructor(e){this.variableNames=[`dy`],this.outputShape=e.inShape;let t=e.filterDepth,n=e.filterHeight,r=e.filterWidth,i=e.strideDepth,a=e.strideHeight,o=e.strideWidth,s=e.dilationDepth,c=e.dilationHeight,l=e.dilationWidth,u=e.effectiveFilterDepth,d=e.effectiveFilterHeight,f=e.effectiveFilterWidth,p=u-1-e.padInfo.front,m=d-1-e.padInfo.top,h=f-1-e.padInfo.left,g=1/(t*n*r);this.userCode=`
      const ivec3 pads = ivec3(${p}, ${m}, ${h});
      const float avgMultiplier = float(${g});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, d) with pos mask(:, :, :, ch) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${u};
            wD += ${s}) {
          float dyD = float(dyDCorner + wD) / ${i}.0;

          if (dyD < 0.0 || dyD >= ${e.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${d};
              wR += ${c}) {
            float dyR = float(dyRCorner + wR) / ${a}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${f};
                wC += ${l}) {
              float dyC = float(dyCCorner + wC) / ${o}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);

              dotProd += dyValue * avgMultiplier;
            }
          }
        }
        setOutput(dotProd);
      }
    `}};function Uk(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,o=a,{filterSize:s,strides:c,pad:l,dimRoundingMode:u}=r,d=new Hk(hs(o.shape,s,c,[1,1,1],l,u));return n.runWebGLProgram(d,[i],o.dtype)}var Wk={kernelName:Xe,backendName:`webgl`,kernelFunc:Uk};function Gk(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,o=a;jC([i,a],`avgPoolGrad`);let{filterSize:s,strides:c,pad:l}=r,u=new Vk(ms(o.shape,s,c,1,l));return n.runWebGLProgram(u,[i],o.dtype)}var Kk={kernelName:Je,backendName:`webgl`,kernelFunc:Gk};function qk(e){let{inputs:t,backend:n,attrs:r}=e,{a:i,b:a}=t,{transposeA:o,transposeB:s}=r;return rk({a:i,b:a,transposeA:o,transposeB:s,backend:n})}var Jk={kernelName:Ze,backendName:`webgl`,kernelFunc:qk},Yk=class{constructor(e,t,n,r,i,a){this.outputShape=[],this.variableNames=[`x`,`mean`,`variance`],W(e,t),W(e,n);let o=`0.0`;r!=null&&(W(e,r),this.variableNames.push(`offset`),o=`getOffsetAtOutCoords()`);let s=`1.0`;i!=null&&(W(e,i),this.variableNames.push(`scale`),s=`getScaleAtOutCoords()`),this.outputShape=e,this.userCode=`
      void main() {
        float x = getXAtOutCoords();
        float mean = getMeanAtOutCoords();
        float variance = getVarianceAtOutCoords();
        float offset = ${o};
        float scale = ${s};
        float inv = scale * inversesqrt(variance + float(${a}));
        setOutput(dot(vec3(x, -mean, offset), vec3(inv, inv, 1)));
      }
    `}},Xk=class{constructor(e,t,n,r,i,a){this.packedInputs=!0,this.packedOutput=!0,this.variableNames=[`x`,`mean`,`variance`],W(e,t),W(e,n);let o=`vec4(0.0)`;r!=null&&(W(e,r),this.variableNames.push(`offset`),o=`getOffsetAtOutCoords()`);let s=`vec4(1.0)`;i!=null&&(W(e,i),this.variableNames.push(`scale`),s=`getScaleAtOutCoords()`),this.outputShape=e,this.userCode=`
      void main() {
        vec4 offset = ${o};
        vec4 scale = ${s};

        vec4 x = getXAtOutCoords();
        vec4 mean = getMeanAtOutCoords();
        vec4 variance = getVarianceAtOutCoords();

        vec4 inv = scale * inversesqrt(variance + vec4(${a}));

        setOutput((x - mean) * inv + offset);
      }
    `}},Zk={kernelName:Ft,backendName:`webgl`,kernelFunc:({inputs:e,backend:t,attrs:n})=>{let{x:r,mean:i,variance:a,offset:o,scale:s}=e;y(i.shape.length===a.shape.length,()=>`Batch normalization gradient requires mean and variance to have equal ranks.`),y(o==null||i.shape.length===o.shape.length,()=>`Batch normalization gradient requires mean and offset to have equal ranks.`),y(s==null||i.shape.length===s.shape.length,()=>`Batch normalization gradient requires mean and scale to have equal ranks.`);let{varianceEpsilon:c}=n;c??=.001;let l=[r,i,a],u=null;o!=null&&(u=o.shape,l.push(o));let d=null;s!=null&&(d=s.shape,l.push(s));let f=O().getBool(`WEBGL_PACK_NORMALIZATION`)?new Xk(r.shape,i.shape,a.shape,u,d,c):new Yk(r.shape,i.shape,a.shape,u,d,c);return t.runWebGLProgram(f,l,l[0].dtype)}},Qk=class{constructor(e){this.variableNames=[`source`],this.outputShape=e,this.rank=e.length;let t=Z(this.rank);this.customUniforms=[{name:`start`,arrayIndex:this.rank,type:`int`}];let n=eA(this.rank),r;r=`
        ${t} sourceLoc;
        ${t} coords = getOutputCoords();
        ${e.map((e,t)=>`sourceLoc.${$k[t]} = start[${t}] + coords.${$k[t]};`).join(`
`)}
      `,this.userCode=`
      void main() {
        ${r}
        setOutput(getSource(${n}));
      }
    `}},$k=[`x`,`y`,`z`,`w`,`u`,`v`];function eA(e){if(e===1)return`sourceLoc`;if(e<=6)return $k.slice(0,e).map(e=>`sourceLoc.`+e).join(`,`);throw Error(`Slicing for rank ${e} is not yet supported`)}var tA=class{constructor(e){this.variableNames=[`source`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.rank=e.length,this.customUniforms=[{name:`start`,arrayIndex:this.rank,type:`int`}];let t=Z(this.rank),n=ND(`coords`,this.rank),r=ND(`sourceLoc`,this.rank),i=this.rank===1?`sourceLoc`:`vec2(${r.slice(-2).join()})`,a=`getChannel(getSource(${r.join()}), ${i})`,o=`
      result.x = ${a};
      if (++${n[this.rank-1]} < ${e[this.rank-1]}) {
        ++${r[this.rank-1]};
        result.y = ${a};
        --${r[this.rank-1]};
      }
    `,s=this.rank===1?``:`
      --${n[this.rank-1]};
      if (++${n[this.rank-2]} < ${e[this.rank-2]}) {
        ++${r[this.rank-2]};
        result.z = ${a};
        if (++${n[this.rank-1]} < ${e[this.rank-1]}) {
          ++${r[this.rank-1]};
          result.w = ${a};
        }
      }
    `,c=this.rank<=4?`sourceLoc = coords +
            ${t}(${e.map((e,t)=>`start[${t}]`).join()});`:e.map((e,t)=>`${r[t]} = ${n[t]} + start[${t}];`).join(`
`);this.userCode=`
      void main() {
        ${t} coords = getOutputCoords();
        ${t} sourceLoc;
        ${c}
        vec4 result = vec4(0.);
        ${o}
        ${s}
        setOutput(result);
      }
    `}};function nA(e,t,n,r){let i=r.texData.get(e.dataId),a=r.makeTensorInfo(n,e.dtype),o=r.texData.get(a.dataId);Object.assign(o,i),o.refCount=1,o.shape=n,o.dtype=e.dtype;let s=av(t,_e(e.shape));i.slice&&(s+=i.slice.flatOffset),o.slice={flatOffset:s,origDataId:i.slice&&i.slice.origDataId||e.dataId};let c=r.dataRefCount.get(o.slice.origDataId)||1;return r.dataRefCount.set(o.slice.origDataId,c+1),a}function rA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{begin:a,size:o}=r,[s,c]=ov(i,a,o);if(K_(i,s,c),S(c)===0)return n.makeTensorInfo(c,i.dtype,[]);if(n.shouldExecuteOnCPU([i])||i.dtype===`string`){let e=_D(n.texData.get(i.dataId).values,s,c,i.shape,i.dtype);return n.makeTensorInfo(c,i.dtype,e)}let{isPacked:l}=n.texData.get(i.dataId),u=iv(i.shape,s,c);if(l||!u){let e=O().getBool(`WEBGL_PACK_ARRAY_OPERATIONS`)?new tA(c):new Qk(c),t=[s];return n.runWebGLProgram(e,[i],i.dtype,t)}return n.uploadToGPU(i.dataId),nA(i,s,c,n)}var iA={kernelName:Wn,backendName:`webgl`,kernelFunc:rA},aA={kernelName:Qe,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockShape:a,crops:o}=r;y(i.shape.length<=4,()=>`batchToSpaceND for rank > 4 with a WebGL backend not implemented yet`);let s=a.reduce((e,t)=>e*t),c=xv(i.shape,a,s),l=Sv(c.length,a.length),u=Cv(i.shape,a,s),d=wv(o,a.length),f=Tv(u,o,a.length),p=[],m=$({inputs:{x:i},backend:n,attrs:{shape:c}}),h=tk({inputs:{x:m},backend:n,attrs:{perm:l}}),g=$({inputs:{x:h},backend:n,attrs:{shape:u}}),_=rA({inputs:{x:g},backend:n,attrs:{begin:d,size:f}});return p.push(m),p.push(h),p.push(g),p.forEach(e=>n.disposeIntermediateTensorInfo(e)),_}};function oA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,weights:a}=t,{size:o}=r,s=RE(n.readSync(i.dataId),n.readSync(a.dataId),a.dtype,a.shape,o);return n.makeTensorInfo([o],a.dtype,s)}var sA={kernelName:$e,backendName:`webgl`,kernelFunc:oA},cA=`
  int r = int(a.r) & int(b.r);
  int g = int(a.g) & int(b.g);
  int rb = int(a.b) & int(b.b);
  int ra = int(a.a) & int(b.a);
  return vec4(r, g, rb, ra);
`,lA=`
  return float(int(a.r) & int(b.r));
`;function uA(e){let{inputs:t,backend:n}=e,{a:r,b:i}=t,a=O().getBool(`WEBGL_PACK_BINARY_OPERATIONS`),o=O().getNumber(`WEBGL_VERSION`);if(n.shouldExecuteOnCPU([r,i])||o===1){let e=n.texData.get(r.dataId).values,t=n.texData.get(i.dataId).values,[a,o]=BE(r.shape,i.shape,e,t,r.dtype),s=n.makeTensorInfo(o,r.dtype),c=n.texData.get(s.dataId);return c.values=a,s}let s;return s=a?new bO(cA,r.shape,i.shape,!1):new vO(lA,r.shape,i.shape),n.runWebGLProgram(s,[r,i],r.dtype)}var dA={kernelName:et,backendName:`webgl`,kernelFunc:uA};function fA(e){let{inputs:t,backend:n}=e,{s0:r,s1:i}=t,a=n.readSync(r.dataId),o=n.readSync(i.dataId),s=W(Array.from(a),Array.from(o));return n.makeTensorInfo([s.length],`int32`,Int32Array.from(s))}var pA={kernelName:tt,backendName:`webgl`,kernelFunc:fA},mA=PO({opSnippet:`return float(a != b);`,cpuKernelImpl:sD,dtype:`bool`}),hA={kernelName:pn,backendName:`webgl`,kernelFunc:mA};function gA(e){let{inputs:t,backend:n}=e,{input:r}=t;return xO({inputs:{x:n.texData.get(r.dataId).complexTensorInfos.real},backend:n})}var _A={kernelName:Dn,backendName:`webgl`,kernelFunc:gA},vA=`return float(int(x));`;function yA(e,t){let n=new GD(e.shape,vA),r=t.runWebGLProgram(n,[e],`int32`);return{dataId:r.dataId,shape:r.shape,dtype:r.dtype}}function bA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{dtype:a}=r;if(a===`complex64`){if(i.dtype===`complex64`)return xO({inputs:{x:i},backend:n});let e=pd(i.shape),t=bA({inputs:{x:i},backend:n,attrs:{dtype:`float32`}}),r=CO({inputs:{real:t,imag:e},backend:n});return e.dispose(),n.disposeIntermediateTensorInfo(t),r}if(i.dtype===`complex64`){let e=gA({inputs:{input:i},backend:n}),t=bA({inputs:{x:e},backend:n,attrs:{dtype:a}});return n.disposeIntermediateTensorInfo(e),t}if(!ce(i.dtype,a)){let e=xO({inputs:{x:i},backend:n});return{dataId:e.dataId,shape:e.shape,dtype:a}}if(n.shouldExecuteOnCPU([i])){let e=n.texData.get(i.dataId).values,[t,r,o]=VE(e,i.shape,i.dtype,a);return n.makeTensorInfo(t,r,o)}if(a===`int32`)return yA(i,n);if(a===`bool`){let e=n.makeTensorInfo([],`bool`,ie(`bool`,1)),t=mA({inputs:{a:i,b:e},backend:n});return n.disposeIntermediateTensorInfo(e),t}throw Error(`Error in Cast: failed to cast ${i.dtype} to ${a}`)}var xA={kernelName:nt,backendName:`webgl`,kernelFunc:bA},SA=`return ceil(x);`,CA={kernelName:rt,backendName:`webgl`,kernelFunc:Q({opSnippet:SA,packedOpSnippet:SA,cpuKernelImpl:HE})},wA=class{constructor(e){this.variableNames=[`A`],this.customUniforms=[{name:`minVal`,type:`float`},{name:`maxVal`,type:`float`}],this.outputShape=e,this.userCode=`

      void main() {
        float value = getAAtOutCoords();
        if (isnan(value)) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, minVal, maxVal));
      }
    `}},TA=class{constructor(e){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`minVal`,type:`float`},{name:`maxVal`,type:`float`}],this.outputShape=e,this.userCode=`
      void main() {
        vec4 value = getAAtOutCoords();

        if (any(isnan(value))) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, vec4(minVal), vec4(maxVal)));
      }
    `}};function EA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{clipValueMin:a,clipValueMax:o}=r,s;s=O().getBool(`WEBGL_PACK_CLIP`)?new TA(i.shape):new wA(i.shape);let c=[[a],[o]];return n.runWebGLProgram(s,[i],i.dtype,c)}var DA={kernelName:it,backendName:`webgl`,kernelFunc:EA},OA=class{constructor(e){this.variableNames=[`real`,`imag`],this.outputShape=e,this.userCode=`
      void main() {
        float re = abs(getRealAtOutCoords());
        float im = abs(getImagAtOutCoords());
        float mx = max(re, im);

        // sadly the length function in glsl is not underflow-safe
        // (at least not on Intel GPUs). So the safe solution is
        // to ensure underflow-safety in all cases.
        setOutput(
          mx == 0.0 ? 0.0 : mx * length(vec2(1, min(re, im)/mx))
        );
      }
    `}};function kA(e,t){return{dataId:t.dataId,dtype:t.dtype,shape:e.shape}}function AA(e){let{inputs:t,backend:n}=e,{x:r}=t,i=n.texData.get(r.dataId),a=new OA(r.shape),o=[kA(r,i.complexTensorInfos.real),kA(r,i.complexTensorInfos.imag)];return n.runWebGLProgram(a,o,o[0].dtype)}var jA={kernelName:ot,backendName:`webgl`,kernelFunc:AA},MA=class{constructor(e){this.outputShape=[],this.outputShape=pv(e,1),this.variableNames=e.map((e,t)=>`T${t}`);let t=Array(e.length-1);t[0]=e[0][1];for(let n=1;n<t.length;n++)t[n]=t[n-1]+e[n][1];let n=[`if (yC < ${t[0]}) setOutput(getT0(yR, yC));`];for(let e=1;e<t.length;e++){let r=t[e-1];n.push(`else if (yC < ${t[e]}) setOutput(getT${e}(yR, yC-${r}));`)}let r=t.length,i=t[t.length-1];n.push(`else setOutput(getT${r}(yR, yC-${i}));`),this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int yR = coords.x;
        int yC = coords.y;

        ${n.join(`
        `)}
      }
    `}},NA=class{constructor(e,t){this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[],this.outputShape=pv(e,t);let n=this.outputShape,r=n.length,i=Z(r),a=ND(`coords`,r),o=[`x`,`y`,`z`,`w`,`u`,`v`].slice(0,r);this.variableNames=e.map((e,t)=>`T${t}`);let s=Array(e.length-1);s[0]=e[0][t];for(let n=1;n<s.length;n++)s[n]=s[n-1]+e[n][t];let c=o[t],l=o.slice(-2),u=o.join(),d=`if (${c} < ${s[0]}) {
        return getChannel(
            getT0(${u}), vec2(${l.join()}));
        }`;for(let e=1;e<s.length;e++){let t=s[e-1];d+=`
        if (${c} < ${s[e]}  && ${c} >= ${s[e-1]}) {
          return getChannel(
            getT${e}(${PA(o,c,t)}),
            vec2(${PA(l,c,t)}));
        }`}let f=s.length,p=s[s.length-1];d+=`
        return getChannel(
          getT${f}(${PA(o,c,p)}),
          vec2(${PA(l,c,p)}));`,this.userCode=`
      float getValue(${o.map(e=>`int `+e)}) {
        ${d}
      }

      void main() {
        ${i} coords = getOutputCoords();
        vec4 result = vec4(getValue(${a}), 0., 0., 0.);

        ${a[r-1]} = ${a[r-1]} + 1;
        if (${a[r-1]} < ${n[r-1]}) {
          result.g = getValue(${a});
        }

        ${a[r-2]} = ${a[r-2]} + 1;
        if (${a[r-2]} < ${n[r-2]}) {
          result.a = getValue(${a});
        }

        ${a[r-1]} = ${a[r-1]} - 1;
        if (${a[r-2]} < ${n[r-2]} &&
            ${a[r-1]} < ${n[r-1]}) {
          result.b = getValue(${a});
        }
        setOutput(result);
      }
    `}};function PA(e,t,n){let r=e.indexOf(t);return e.map((e,t)=>t===r?`${e} - ${n}`:e).join()}function FA(e){let{inputs:t,backend:n}=e,{input:r}=t;return xO({inputs:{x:n.texData.get(r.dataId).complexTensorInfos.imag},backend:n})}var IA={kernelName:Ht,backendName:`webgl`,kernelFunc:FA};function LA(e,t,n){let r=e[0].dtype;if(r===`complex64`){let r=e.map(e=>gA({inputs:{input:e},backend:n})),i=e.map(e=>FA({inputs:{input:e},backend:n})),a=LA(r,t,n),o=LA(i,t,n),s=CO({inputs:{real:a,imag:o},backend:n});return r.forEach(e=>n.disposeIntermediateTensorInfo(e)),i.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.disposeIntermediateTensorInfo(a),n.disposeIntermediateTensorInfo(o),s}let i=n.shouldExecuteOnCPU(e);if(r===`string`&&(i=!0),i){let i=e.map(e=>{let r=[-1,S(e.shape.slice(t))];return $({inputs:{x:e},backend:n,attrs:{shape:r}})}),a=UE(i.map(e=>({vals:n.readSync(e.dataId),shape:e.shape})),pv(i.map(e=>e.shape),1),r,i[0].shape[0]===1),o=pv(e.map(e=>e.shape),t),s=n.makeTensorInfo(o,r,a);return i.forEach(e=>n.disposeIntermediateTensorInfo(e)),s}let a=e.filter(e=>S(e.shape)>0),o=O().getBool(`WEBGL_PACK_ARRAY_OPERATIONS`)&&a[0].shape.length>1;if(a.length===1){let t=o?new GD(e[0].shape,QD):new aO(e[0].shape,QD);return n.runWebGLProgram(t,e,r)}let s=O().getNumber(`WEBGL_MAX_TEXTURES_IN_SHADER`);if(a.length>s){let e=[];for(let r=0;r<a.length;r+=s){let i=a.slice(r,r+s);e.push(LA(i,t,n))}let r=LA(e,t,n);for(let t of e)n.disposeIntermediateTensorInfo(t);return r}if(o){let e=new NA(a.map(e=>e.shape),t);return n.runWebGLProgram(e,a,r)}let{tensors2D:c,outShape:l}=RA(a,t,n),u=new MA(c.map(e=>e.shape)),d=n.runWebGLProgram(u,c,r);c.forEach(e=>n.disposeIntermediateTensorInfo(e));let f=$({inputs:{x:d},attrs:{shape:l},backend:n});return n.disposeIntermediateTensorInfo(d),f}function RA(e,t,n){let r=pv(e.map(e=>e.shape),t);return{tensors2D:e.map(e=>$({inputs:{x:e},attrs:{shape:[-1,S(e.shape.slice(t))]},backend:n})),outShape:r}}function zA(e){let{inputs:t,backend:n,attrs:r}=e,{axis:i}=r,a=D(i,t[0].shape)[0];fv(t.map(e=>e.shape),a);let o=pv(t.map(e=>e.shape),a);if(S(o)===0)return n.makeTensorInfo(o,t[0].dtype,[]);let s=t.filter(e=>S(e.shape)>0);return s.length===1?xO({inputs:{x:s[0]},backend:n}):LA(s,a,n)}var BA={kernelName:st,backendName:`webgl`,kernelFunc:zA},VA=class{constructor(e,t=!1,n=null,r=!1,i=!1){this.variableNames=[`x`,`W`],this.outputShape=e.outShape;let a=e.padInfo.top,o=e.padInfo.left,s=e.strideHeight,c=e.strideWidth,l=e.dilationHeight,u=e.dilationWidth,d=e.filterHeight,f=e.filterWidth,p=Math.floor(e.inChannels/4)*4,m=e.inChannels%4,h=e.dataFormat===`channelsLast`,g=h?1:2,_=h?2:3,v=h?3:1,y=``,b=``;n&&(y=r?`float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${n}
        }`:i?`float activation(float a) {
          float b = getLeakyreluAlphaAtOutCoords();
          ${n}
        }`:`
          float activation(float x) {
            ${n}
          }
        `,b=`result = activation(result);`);let x=t?`result += getBiasAtOutCoords();`:``;t&&this.variableNames.push(`bias`),r&&this.variableNames.push(`preluActivationWeights`),i&&this.variableNames.push(`leakyreluAlpha`),this.userCode=`
      ${y}

      const ivec2 strides = ivec2(${s}, ${c});
      const ivec2 pads = ivec2(${a}, ${o});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d2 = coords[${v}];

        ivec2 xRCCorner =
            ivec2(coords[${g}], coords[${_}]) * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${d}; wR++) {
          int xR = xRCorner + wR * ${l};

          if (xR < 0 || xR >= ${e.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${f}; wC++) {
            int xC = xCCorner + wC * ${u};

            if (xC < 0 || xC >= ${e.inWidth}) {
              continue;
            }

            for (int d1 = 0; d1 < ${p}; d1 += 4) {
              vec4 wValues = vec4(
                getW(wR, wC, d1, d2),
                getW(wR, wC, d1 + 1, d2),
                getW(wR, wC, d1 + 2, d2),
                getW(wR, wC, d1 + 3, d2)
              );

              if (${h}) {
                vec4 xValues = vec4(
                  getX(batch, xR, xC, d1),
                  getX(batch, xR, xC, d1 + 1),
                  getX(batch, xR, xC, d1 + 2),
                  getX(batch, xR, xC, d1 + 3)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec4 xValues = vec4(
                  getX(batch, d1, xR, xC),
                  getX(batch, d1 + 1, xR, xC),
                  getX(batch, d1 + 2, xR, xC),
                  getX(batch, d1 + 3, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }
            }

            if (${m===1}) {

              if (${h}) {
                dotProd +=
                    getX(batch, xR, xC, ${p}) *
                    getW(wR, wC, ${p}, d2);
              } else {
                dotProd +=
                    getX(batch, ${p}, xR, xC) *
                    getW(wR, wC, ${p}, d2);
              }

            } else if (${m===2}) {
              vec2 wValues = vec2(
                getW(wR, wC, ${p}, d2),
                getW(wR, wC, ${p} + 1, d2)
              );

              if (${h}) {
                vec2 xValues = vec2(
                  getX(batch, xR, xC, ${p}),
                  getX(batch, xR, xC, ${p} + 1)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec2 xValues = vec2(
                  getX(batch, ${p}, xR, xC),
                  getX(batch, ${p} + 1, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            } else if (${m===3}) {
              vec3 wValues = vec3(
                getW(wR, wC, ${p}, d2),
                getW(wR, wC, ${p} + 1, d2),
                getW(wR, wC, ${p} + 2, d2)
              );

              if (${h}) {
                vec3 xValues = vec3(
                  getX(batch, xR, xC, ${p}),
                  getX(batch, xR, xC, ${p} + 1),
                  getX(batch, xR, xC, ${p} + 2)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec3 xValues = vec3(
                  getX(batch, ${p}, xR, xC),
                  getX(batch, ${p} + 1, xR, xC),
                  getX(batch, ${p} + 2, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            }
          }
        }

        float result = dotProd;
        ${x}
        ${b}
        setOutput(result);
      }
    `}},HA=class{constructor(e){this.variableNames=[`x`,`W`],this.outputShape=e.outShape;let t=e.padInfo.front,n=e.padInfo.top,r=e.padInfo.left,i=e.strideDepth,a=e.strideHeight,o=e.strideWidth,s=e.dilationDepth,c=e.dilationHeight,l=e.dilationWidth,u=e.filterDepth,d=e.filterHeight,f=e.filterWidth,p=Math.floor(e.inChannels/4)*4,m=e.inChannels%4;this.userCode=`
      const ivec3 strides = ivec3(${i}, ${a}, ${o});
      const ivec3 pads = ivec3(${t}, ${n}, ${r});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d2 = coords.u;

        ivec3 xFRCCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xFCorner = xFRCCorner.x;
        int xRCorner = xFRCCorner.y;
        int xCCorner = xFRCCorner.z;

        // Convolve x(?, ?, ?, d1) with w(:, :, :, d1, d2) to get
        // y(yF, yR, yC, d2). ? = to be determined. : = across all
        // values in that axis.
        float dotProd = 0.0;
        for (int wF = 0; wF < ${u}; wF++) {
          int xF = xFCorner + wF * ${s};

          if (xF < 0 || xF >= ${e.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${d}; wR++) {
            int xR = xRCorner + wR * ${c};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${f}; wC++) {
              int xC = xCCorner + wC * ${l};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              for (int d1 = 0; d1 < ${p}; d1 += 4) {
                vec4 xValues = vec4(
                  getX(batch, xF, xR, xC, d1),
                  getX(batch, xF, xR, xC, d1 + 1),
                  getX(batch, xF, xR, xC, d1 + 2),
                  getX(batch, xF, xR, xC, d1 + 3)
                );
                vec4 wValues = vec4(
                  getW(wF, wR, wC, d1, d2),
                  getW(wF, wR, wC, d1 + 1, d2),
                  getW(wF, wR, wC, d1 + 2, d2),
                  getW(wF, wR, wC, d1 + 3, d2)
                );

                dotProd += dot(xValues, wValues);
              }

              if (${m===1}) {
                dotProd +=
                  getX(batch, xF, xR, xC, ${p}) *
                  getW(wF, wR, wC, ${p}, d2);
              } else if (${m===2}) {
                vec2 xValues = vec2(
                  getX(batch, xF, xR, xC, ${p}),
                  getX(batch, xF, xR, xC, ${p} + 1)
                );
                vec2 wValues = vec2(
                  getW(wF, wR, wC, ${p}, d2),
                  getW(wF, wR, wC, ${p} + 1, d2)
                );
                dotProd += dot(xValues, wValues);
              } else if (${m===3}) {
                vec3 xValues = vec3(
                  getX(batch, xF, xR, xC, ${p}),
                  getX(batch, xF, xR, xC, ${p} + 1),
                  getX(batch, xF, xR, xC, ${p} + 2)
                );
                vec3 wValues = vec3(
                  getW(wF, wR, wC, ${p}, d2),
                  getW(wF, wR, wC, ${p} + 1, d2),
                  getW(wF, wR, wC, ${p} + 2, d2)
                );
                dotProd += dot(xValues, wValues);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}},UA=class{constructor(e,t=!1,n=null,r=!1,i=!1){this.variableNames=[`x`,`W`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`pads`,type:`ivec2`},{name:`strides`,type:`ivec2`},{name:`dilations`,type:`ivec2`},{name:`inDims`,type:`ivec2`}],this.outputShape=e.outShape,this.enableShapeUniforms=Iw(this.outputShape.length);let a=e.padInfo.left,o=e.strideWidth,s=e.dilationWidth,c=e.filterHeight,l=e.filterWidth,u=l,d=`
       int xR; int xC; int xCOffset;
       vec4 wTexel; vec4 previous; vec4 final;`;for(let e=0;e<l;e++)d+=`
           vec4 xTexelC${e*2};
           int xTexelC${e*2}Ready;
           vec4 xTexelC${e*2+1};
           int xTexelC${e*2+1}Ready;
           vec4 xC${e};`;d+=`
     for (int r = 0; r < ${c}; r++) {
      for (int d1 = 0; d1 < ${e.inChannels}; d1 += 2) {
       `;for(let e=0;e<l;e++)d+=`
           xTexelC${e*2} = vec4(0.0);
           xTexelC${e*2}Ready = 0;
           xTexelC${e*2+1} = vec4(0.0);
           xTexelC${e*2+1}Ready = 0;
           xC${e} = vec4(0.0);`;d+=`
         xR = xRCorner + r * dilations[0];
         if (xR >=0 && xR < inDims[0]) {
       `;for(let t=0;t<(u+1)/2;t++){let n=t*2;if(d+=`
           xC = xCCorner + ${n*s};
           `,o===1){if(n<l&&(a%2==1?(d+=`
                 xCOffset = xC + 1;
                 if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${n}Ready == 0) {
                   xTexelC${n} = getX(batch, xR, xCOffset, d1);

                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${n}.zw = vec2(0.0);
                   }
                   xTexelC${n}Ready = 1;
                 }
               `,s===1&&n>0?d+=`
                 xC${n} = vec4(xTexelC${n-2}.zw, xTexelC${n}.xy);
                 `:d+=`
                   xCOffset = xC + 1 - 2;

                   if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);

                     // Need to manually clear unused channels in case
                     // we're reading from recycled texture.
                     if (xCOffset + 1 >= inDims[1]) {
                       previous.zw = vec2(0.0);
                     }

                     xC${n} = vec4(previous.zw, xTexelC${n}.xy);
                   } else {
                     xC${n} = vec4(0.0, 0.0, xTexelC${n}.xy);
                   }
                   `):d+=`
                 if (xC >= 0 && xC < inDims[1] && xTexelC${n}Ready == 0) {
                   xTexelC${n} = getX(batch, xR, xC, d1);
                   if (xC + 1 >= inDims[1]) {
                     xTexelC${n}.zw = vec2(0.0);
                   }
                   xTexelC${n}Ready = 1;
                 }

                 xC${n} = xTexelC${n};
                 `,n+1<l)){let e=a%2==0?g(s):s;s%2==0&&a%2==1||s%2!=0&&a%2!=1?(d+=`
                   xCOffset = xC + imod(pads[1], 2) + ${e};

                   if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${n+1}Ready == 0) {
                     xTexelC${n+1} = getX(batch, xR, xCOffset, d1);

                     // Need to manually clear unused channels in case
                     // we're reading from recycled texture.
                     if (xCOffset + 1 >= inDims[1]) {
                       xTexelC${n+1}.zw = vec2(0.0);
                     }
                     xTexelC${n+1}Ready = 1;
                   }
                   `,s>1?d+=`
                     xCOffset -= 2;
                     if (xCOffset >= 0 && xCOffset < inDims[1]) {
                      previous = getX(batch, xR, xCOffset, d1);
                      xC${n+1} = vec4(previous.zw, xTexelC${n+1}.xy);
                     } else {
                      xC${n+1} = vec4(0.0, 0.0, xTexelC${n+1}.xy);
                     }
                     `:d+=`
                     xC${n+1} = vec4(xTexelC${n}.zw, xTexelC${n+1}.xy);
                     `):e===1?d+=`
                     xC${n+1} = xTexelC${n};
                     `:d+=`
                     xCOffset = xC + ${e};

                     if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${n+1}Ready == 0) {
                       xTexelC${n+1} = getX(batch, xR, xCOffset, d1);
                       if (xCOffset + 1 >= inDims[1]) {
                         xTexelC${n+1}.zw = vec2(0.0);
                       }
                       xTexelC${n+1}Ready = 1;
                     }

                     xC${n+1} = xTexelC${n+1};
                     `}}else n<l&&(a%2==1?(d+=`
                 xCOffset = xC + 1 - strides[1];
                 if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${n}Ready == 0) {
                   xTexelC${n} = getX(batch, xR, xCOffset, d1);
                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${n}.zw = vec2(0.0);
                   }
                   xTexelC${n}Ready = 1;
                 }

                 if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${n+1}Ready == 0) {
                   xTexelC${n+1} = getX(batch, xR, xC + 1, d1);
                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xC + 2 >= inDims[1]) {
                     xTexelC${n+1}.zw = vec2(0.0);
                   }
                   xTexelC${n+1}Ready = 1;
                 }

                 xC${n} = vec4(xTexelC${n}.zw, xTexelC${n+1}.zw);
               `,n+1<l&&(d+=`
                   final = vec4(0.0);
                   xCOffset = xC + 1 + strides[1];
                   if(xCOffset >= 0 && xCOffset < inDims[1]) {
                     final = getX(batch, xR, xCOffset, d1);
                   }
                   xC${n+1} = vec4(xTexelC${n+1}.xy, final.xy);
                 `)):(d+=`
                 if(xC >= 0 && xC < inDims[1] && xTexelC${n}Ready == 0) {
                   xTexelC${n} = getX(batch, xR, xC, d1);
                   if (xC + 1 >= inDims[1]) {
                     xTexelC${n}.zw = vec2(0.0);
                   }
                   xTexelC${n}Ready = 1;
                 }

                 xCOffset = xC + strides[1];
                 if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${n+1}Ready == 0) {
                   xTexelC${n+1} = getX(batch, xR, xCOffset, d1);
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${n+1}.zw = vec2(0.);
                   }
                   xTexelC${n+1}Ready = 1;
                 }

                 xC${n} = vec4(
                   xTexelC${n}.xy, xTexelC${n+1}.xy);
               `,n+1<l&&(d+=`
                   xC${n+1} = vec4(xTexelC${n}.zw, xTexelC${n+1}.zw);
                 `)));n<l&&(d+=`
             wTexel = getW(r, ${n}, d1, d2);
             dotProd += xC${n}.xxzz * vec4(wTexel.xy, wTexel.xy);
             if(d1 + 1 < ${e.inChannels}) {
               dotProd += xC${n}.yyww * vec4(wTexel.zw, wTexel.zw);
             }
           `,n+1<l&&(d+=`
               wTexel = getW(r, ${n+1}, d1, d2);
               dotProd += xC${n+1}.xxzz * vec4(wTexel.xy, wTexel.xy);
               if(d1 + 1 < ${e.inChannels}) {
                 dotProd += xC${n+1}.yyww * vec4(wTexel.zw, wTexel.zw);
               }
             `))}d+=`
     }
   `,d+=`
     }
   `,d+=`
     }
   `;let f=``,p=``;n&&(f=r?`vec4 activation(vec4 a) {
           vec4 b = getPreluActivationWeightsAtOutCoords();
           ${n}
         }`:i?`vec4 activation(vec4 a) {
           vec4 b = getLeakyreluAlphaAtOutCoords();
           ${n}
         }`:`vec4 activation(vec4 x) {
           ${n}
         }`,p=`result = activation(result);`);let m=t?`result += getBiasAtOutCoords();`:``;t&&this.variableNames.push(`bias`),r&&this.variableNames.push(`preluActivationWeights`),i&&this.variableNames.push(`leakyreluAlpha`),this.userCode=`
       ${f}

       void main() {
         ivec4 coords = getOutputCoords();
         int batch = coords.x;
         ivec2 xRCCorner = coords.yz * strides - pads;
         int d2 = coords.w;
         int xRCorner = xRCCorner.x;
         int xCCorner = xRCCorner.y;

         //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
         vec4 dotProd = vec4(0.000000000000001);

         ${d}

         vec4 result = dotProd - vec4(0.000000000000001);
         ${m}
         ${p}
         setOutput(result);
       }
     `}},WA=class{constructor(e,t){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`inputShape`,type:`ivec4`},{name:`pad`,type:`ivec2`},{name:`stride`,type:`ivec2`},{name:`dilation`,type:`ivec2`},{name:`inChannels`,type:`int`},{name:`itemsPerBlockRow`,type:`int`},{name:`outWidth`,type:`int`}],this.outputShape=e,this.enableShapeUniforms=Iw(this.outputShape.length);let{dataFormat:n}=t,r=MC(),i=n===`channelsLast`,a=i?1:2,o=i?2:3,s=this.enableShapeUniforms?`if(blockIndex < outShape[2] && pos < outShape[1]) {`:`if(blockIndex < ${e[2]} && pos < ${e[1]}) {`,c=``;for(let e=0;e<=1;e++)for(let t=0;t<=1;t++)c+=`
          blockIndex = rc.z + ${t};
          pos = rc.y + ${e};

          ${s}
            offsetY = int(blockIndex / outWidth) * stride[0] - pad[0];
            d0 = offsetY + dilation[0] * (pos / itemsPerBlockRow);

            if(d0 < inputShape[${a}] && d0 >= 0) {
              // Use custom imod instead mod. On Intel GPU, mod may generate
              // unexpected value.
              // https://github.com/tensorflow/tfjs/issues/5447
              offsetX = imod(blockIndex, outWidth) * stride[1] - pad[1];
              d1 = offsetX + dilation[1] * (imod(pos, itemsPerBlockRow) /
                  inChannels);

              if(d1 < inputShape[${o}] && d1 >= 0) {

                ch = imod(pos, inChannels);

                if (${i}) {
                  innerDims = vec2(d1, ch);
                  result[${e*2+t}] = getChannel(
                    getA(rc.x, d0, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                } else {
                  innerDims = vec2(d0, d1);
                  result[${e*2+t}] = getChannel(
                    getA(rc.x, ch, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                }
              }
            }
          }
        `;this.userCode=`
      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0);

        int blockIndex, pos, offsetY, d0, offsetX, d1, ch;
        vec2 innerDims;

        ${c}

        ${r.output} = result;
      }
    `}};function GA(e,t){let n=e.length;return n>=3?t?[...e.slice(0,-3),e[n-3]*e[n-2],e[n-1]]:[...e.slice(0,-3),e[n-3],e[n-2]*e[n-1]]:!t&&n===1&&e[0]>1?[e[0],1]:null}function KA({x:e,filter:t,convInfo:n,backend:r,bias:i=null,preluActivationWeights:a=null,leakyreluAlpha:o=0,activation:s=null}){let c=e.shape,l=r.texData.get(e.dataId),u=n.inChannels,d=c[0]*c[1]*c[2],f=n.outChannels,p=n.dataFormat===`channelsLast`,m,h=[];if(a!=null){let e=GA(a.shape,p);e!=null&&(a=$({inputs:{x:a},backend:r,attrs:{shape:e}}),h.push(a))}if(i!=null){let e=GA(i.shape,p);e!=null&&(i=$({inputs:{x:i},backend:r,attrs:{shape:e}}),h.push(i))}if(!((d===1||f===1)&&u>1e3)&&l.isPacked&&p&&l.texture!=null&&c[2]%2!=0&&w(l.shape.slice(-3),c.slice(-3))){let u=c[0]*c[1]*(c[2]+1),d={dataId:e.dataId,shape:[1,u,n.inChannels],dtype:e.dtype},f=l.shape;l.shape=l.shape.slice(),l.shape[l.shape.length-2]++,y(vC(l.shape,d.shape),()=>`packed reshape ${l.shape} to ${d.shape} isn't free`);let p=$({inputs:{x:t},backend:r,attrs:{shape:[1,n.inChannels,n.outChannels]}});h.push(p);let g=rk({a:d,b:p,backend:r,transposeA:!1,transposeB:!1,bias:i,activation:s,preluActivationWeights:a,leakyreluAlpha:o}),_=r.texData.get(g.dataId);y(_.isPacked,()=>`batchMatMul result is expected to be packed`),l.shape=f,_.shape=n.outShape,m=xO({inputs:{x:g},backend:r}),m.shape=n.outShape,h.push(g)}else{let c=n.outHeight*n.outWidth,l=$({inputs:{x:e},backend:r,attrs:{shape:p?[n.batchSize,c,n.inChannels]:[n.batchSize,n.inChannels,c]}}),u=$({inputs:{x:t},backend:r,attrs:{shape:[1,n.inChannels,n.outChannels]}}),d=rk({a:p?l:u,b:p?u:l,transposeA:!p,transposeB:!1,backend:r,bias:i,activation:s,preluActivationWeights:a,leakyreluAlpha:o});m=$({inputs:{x:d},backend:r,attrs:{shape:n.outShape}}),h.push(l),h.push(u),h.push(d)}for(let e of h)r.disposeIntermediateTensorInfo(e);return m}function qA({x:e,filter:t,convInfo:n,backend:r,bias:i=null,preluActivationWeights:a=null,leakyreluAlpha:o=0,activation:s=null}){let{filterWidth:c,filterHeight:l,inChannels:u,outWidth:d,outHeight:f,dataFormat:p}=n,m=p===`channelsLast`,h=c*l*u,g=f*d,_=[n.batchSize,h,g],v=[];if(a!=null){let e=GA(a.shape,m);e!=null&&(a=$({inputs:{x:a},backend:r,attrs:{shape:e}}),v.push(a))}if(i!=null){let e=GA(i.shape,m);e!=null&&(i=$({inputs:{x:i},backend:r,attrs:{shape:e}}),v.push(i))}let y=$({inputs:{x:t},backend:r,attrs:{shape:[1,h,S(t.shape)/h]}});v.push(y);let b=new WA(_,n),x=[e.shape,[n.padInfo.top,n.padInfo.left],[n.strideHeight,n.strideWidth],[n.dilationHeight,n.dilationWidth],[n.inChannels],[n.filterWidth*n.inChannels],[n.outWidth]],C=r.runWebGLProgram(b,[e],`float32`,x),w=$({inputs:{x:C},backend:r,attrs:{shape:_}});v.push(C),v.push(w);let ee=i!=null,T=a!=null,E=s===`leakyrelu`,te=s?FO(s,!0):null,ne=new IO(m?w.shape:y.shape,m?y.shape:w.shape,m?[n.batchSize,g,n.outChannels]:[n.batchSize,n.outChannels,g],!0,!1,ee,te,T,E),D=m?[w,y]:[y,w];if(i&&D.push(i),T&&D.push(a),E){let e=r.makeTensorInfo([],`float32`,$r(o,`float32`));D.push(e),v.push(e)}let re=r.runWebGLProgram(ne,D,`float32`),ie=$({inputs:{x:re},backend:r,attrs:{shape:n.outShape}});v.push(re);for(let e of v)r.disposeIntermediateTensorInfo(e);return ie}function JA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dataFormat:c,dilations:l,dimRoundingMode:u}=r,d=As(c),f=gs(i.shape,a.shape,o,l,s,u,!1,d),p;if(f.filterHeight===1&&f.filterWidth===1&&f.dilationHeight===1&&f.dilationWidth===1&&f.strideHeight===1&&f.strideWidth===1&&(f.padInfo.type===`SAME`||f.padInfo.type===`VALID`))p=KA({x:i,filter:a,convInfo:f,backend:n});else if(f.strideWidth<=2&&d===`channelsLast`&&O().getBool(`WEBGL_EXP_CONV`)){let e=new UA(f),t=[[f.padInfo.top,f.padInfo.left],[f.strideHeight,f.strideWidth],[f.dilationHeight,f.dilationWidth],[f.inHeight,f.inWidth]];p=n.runWebGLProgram(e,[i,a],`float32`,t)}else if(O().getBool(`WEBGL_CONV_IM2COL`))p=qA({x:i,filter:a,convInfo:f,backend:n});else{let e=new VA(f);p=n.runWebGLProgram(e,[i,a],`float32`)}let m=$({inputs:{x:p},backend:n,attrs:{shape:f.outShape}});return n.disposeIntermediateTensorInfo(p),m}var YA={kernelName:ct,backendName:`webgl`,kernelFunc:JA},XA=class{constructor(e){this.variableNames=[`x`,`dy`],this.outputShape=e.filterShape;let t=e.strideHeight,n=e.strideWidth,r=e.padInfo.top,i=e.padInfo.left,a=e.dataFormat===`channelsLast`;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int d2 = coords.w;

        // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yR = 0; yR < ${e.outHeight}; yR++) {
            int xR = wR + yR * ${t} - ${r};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${e.outWidth}; yC++) {
              int xC = wC + yC * ${n} - ${i};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              ${a?`float dyValue = getDy(b, yR, yC, d2);
              float xValue = getX(b, xR, xC, d1);
              dotProd += (xValue * dyValue);`:`float dyValue = getDy(b, d2, yR, yC);
              float xValue = getX(b, d1, xR, xC);
              dotProd += (xValue * dyValue);`}
            }
          }
        }
        setOutput(dotProd);
      }
    `}},ZA=class{constructor(e){this.variableNames=[`dy`,`W`],this.outputShape=e.inShape;let t=e.filterHeight,n=e.filterWidth,r=e.strideHeight,i=e.strideWidth,a=e.dataFormat===`channelsLast`,o=t-1-e.padInfo.top,s=n-1-e.padInfo.left,c=a?1:2,l=a?2:3,u=a?3:1;this.userCode=`
      const ivec2 pads = ivec2(${o}, ${s});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[${u}];

        ivec2 dyCorner = ivec2(coords[${c}], coords[${l}]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / ${r}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${n}; wC++) {
            float dyC = float(dyCCorner + wC) / ${i}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${n} - 1 - wC;

            for (int d2 = 0; d2 < ${e.outChannels}; d2++) {

              if (${a}) {
                float xValue = getDy(batch, idyR, idyC, d2);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              } else {
                float xValue = getDy(batch, d2, idyR, idyC);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }

            }
          }
        }
        setOutput(dotProd);
      }
    `}},QA=class{constructor(e){this.variableNames=[`x`,`dy`],this.outputShape=e.filterShape;let t=e.strideDepth,n=e.strideHeight,r=e.strideWidth,i=e.padInfo.front,a=e.padInfo.top,o=e.padInfo.left;this.userCode=`
      void main() {
        ivec5 coords = getOutputCoords();
        int wF = coords.x;
        int wR = coords.y;
        int wC = coords.z;
        int d1 = coords.w;
        int d2 = coords.u;

        float dotProd = 0.0;

        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yF = 0; yF < ${e.outDepth}; yF++) {
            int xF = wF + yF * ${t} - ${i};

            if (xF < 0 || xF >= ${e.inDepth}) {
              continue;
            }

            for (int yR = 0; yR < ${e.outHeight}; yR++) {
              int xR = wR + yR * ${n} - ${a};

              if (xR < 0 || xR >= ${e.inHeight}) {
                continue;
              }

              for (int yC = 0; yC < ${e.outWidth}; yC++) {
                int xC = wC + yC * ${r} - ${o};

                if (xC < 0 || xC >= ${e.inWidth}) {
                  continue;
                }

                float dyValue = getDy(b, yF, yR, yC, d2);
                float xValue = getX(b, xF, xR, xC, d1);
                dotProd += (xValue * dyValue);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}},$A=class{constructor(e){this.variableNames=[`dy`,`W`],this.outputShape=e.inShape;let t=e.filterDepth,n=e.filterHeight,r=e.filterWidth,i=e.strideDepth,a=e.strideHeight,o=e.strideWidth,s=t-1-e.padInfo.front,c=n-1-e.padInfo.top,l=r-1-e.padInfo.left;this.userCode=`
      const ivec3 pads = ivec3(${s}, ${c}, ${l});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.u;


        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyFCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        float dotProd = 0.0;
        for (int wF = 0; wF < ${t}; wF++) {
          float dyF = float(dyFCorner + wF) / ${i}.0;

          if (dyF < 0.0 || dyF >= ${e.outDepth}.0 || fract(dyF) > 0.0) {
            continue;
          }
          int idyF = int(dyF);

          int wFPerm = ${t} - 1 - wF;

          for (int wR = 0; wR < ${n}; wR++) {
            float dyR = float(dyRCorner + wR) / ${a}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
              fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            int wRPerm = ${n} - 1 - wR;

            for (int wC = 0; wC < ${r}; wC++) {
              float dyC = float(dyCCorner + wC) / ${o}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              int wCPerm = ${r} - 1 - wC;

              for (int d2 = 0; d2 < ${e.outChannels}; d2++) {
                float xValue = getDy(batch, idyF, idyR, idyC, d2);
                float wValue = getW(wFPerm, wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}};function ej(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,pad:s,dataFormat:c,dimRoundingMode:l,filterShape:u}=r,d=As(c),f=new XA(gs(i.shape,u,o,1,s,l,!1,d));return n.runWebGLProgram(f,[i,a],`float32`)}var tj={kernelName:lt,backendName:`webgl`,kernelFunc:ej},nj=class{constructor(e){this.variableNames=[`dy`,`W`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`strides`,type:`vec2`}],this.outputShape=e.inShape,this.enableShapeUniforms=Iw(this.outputShape.length);let t=e.filterHeight,n=e.filterWidth,r=t-1-e.padInfo.top,i=n-1-e.padInfo.left;this.userCode=`
      const ivec2 pads = ivec2(${r}, ${i});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[3];

        ivec2 dyCorner = ivec2(coords[1], coords[2]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        vec4 result = vec4(0.);
        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / strides[0];
          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);
          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${n}; wC++) {
            int wCPerm = ${n} - 1 - wC;

            float dyC = float(dyCCorner + wC) / strides[1];
            bool idyCVal = (dyC >= 0.0) && (dyC < ${e.outWidth}.0)
              && (fract(dyC) == 0.0);
            int idyC = int(dyC);

            float dyC2 = float(dyCCorner + wC + 1) / strides[1];
            bool idyCVal2 = (dyC2 >= 0.0) && (dyC2 < ${e.outWidth}.0)
              && (fract(dyC2) == 0.0);
            int idyC2 = int(dyC2);

            if (idyCVal && idyCVal2) {
              for (int d2 = 0; d2 < ${e.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC, d2);
                vec4 dySample2 = (idyC / 2 == idyC2 / 2) ?
                  dySample : getDy(batch, idyR, idyC2, d2);

                vec2 dyValue = mod(float(idyC), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.xy += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));

                dyValue = mod(float(idyC2), 2.) == 0. ?
                  dySample2.xy : dySample2.zw;
                result.zw += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            } else if (idyCVal) {
              for (int d2 = 0; d2 < ${e.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC, d2);
                vec2 dyValue = mod(float(idyC), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.xy += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            } else if (idyCVal2) {
              for (int d2 = 0; d2 < ${e.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC2, d2);
                vec2 dyValue = mod(float(idyC2), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.zw += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            }
          }
        }
        setOutput(result);
      }
    `}};function rj(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{inputShape:o,strides:s,pad:c,dataFormat:l,dimRoundingMode:u}=r,d=As(l),f=gs(o,a.shape,s,1,c,u,!1,d);if(O().getBool(`WEBGL_PACK_CONV2DTRANSPOSE`)&&d===`channelsLast`){let e=[[f.strideHeight,f.strideWidth]],t=new nj(f);return n.runWebGLProgram(t,[i,a],`float32`,e)}else{let e=new ZA(f);return n.runWebGLProgram(e,[i,a],`float32`)}}var ij={kernelName:ut,backendName:`webgl`,kernelFunc:rj};function aj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c}=r,l=new HA(_s(i.shape,a.shape,o,c,s));return n.runWebGLProgram(l,[i,a],`float32`)}var oj={kernelName:dt,backendName:`webgl`,kernelFunc:aj};function sj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,pad:s,filterShape:c}=r,l=new QA(_s(i.shape,c,o,1,s));return n.runWebGLProgram(l,[i,a],`float32`)}var cj={kernelName:ft,backendName:`webgl`,kernelFunc:sj};function lj(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{pad:o,strides:s,inputShape:c}=r,l=new $A(_s(c,a.shape,s,1,o));return n.runWebGLProgram(l,[i,a],`float32`)}var uj={kernelName:pt,backendName:`webgl`,kernelFunc:lj},dj={kernelName:`Cos`,backendName:`webgl`,kernelFunc:Q({opSnippet:NO+`
  return cos(x);
`,packedOpSnippet:`
  vec4 result = cos(x);
  bvec4 isNaN = isnan(x);
  ${yO}
  return result;
`})},fj={kernelName:mt,backendName:`webgl`,kernelFunc:Q({opSnippet:`
  float e2x = exp(-x);
  return (e2x + 1.0 / e2x) / 2.0;
`})},pj=class{constructor(e,t,n,r,i){this.variableNames=[`Image`,`Boxes`,`BoxInd`],this.outputShape=[];let[a,o,s,c]=e,[l]=t,[u,d]=n;this.outputShape=[l,u,d,c];let f=+(r===`bilinear`),[p,m]=[`${o-1}.0`,`${s-1}.0`],[h,g,_]=u>1?[`${(o-1)/(u-1)}`,`(y2-y1) * height_ratio`,`y1*${p} + float(y)*(height_scale)`]:[`0.0`,`0.0`,`0.5 * (y1+y2) * ${p}`],[v,y,b]=d>1?[`${(s-1)/(d-1)}`,`(x2-x1) * width_ratio`,`x1*${m} + float(x)*(width_scale)`]:[`0.0`,`0.0`,`0.5 * (x1+x2) * ${m}`];this.userCode=`
      const float height_ratio = float(${h});
      const float width_ratio = float(${v});
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int y = coords[1];
        int x = coords[2];
        int d = coords[3];

        // get box vals
        float y1 = getBoxes(b,0);
        float x1 = getBoxes(b,1);
        float y2 = getBoxes(b,2);
        float x2 = getBoxes(b,3);

        // get image in batch index
        int bInd = round(getBoxInd(b));
        if(bInd < 0 || bInd >= ${a}) {
          return;
        }

        float height_scale = ${g};
        float width_scale = ${y};

        float in_y = ${_};
        if( in_y < 0.0 || in_y > ${p} ) {
          setOutput(float(${i}));
          return;
        }
        float in_x = ${b};
        if( in_x < 0.0 || in_x > ${m} ) {
          setOutput(float(${i}));
          return;
        }

        vec2 sourceFracIndexCR = vec2(in_x,in_y);
        if(${f} == 1) {
          // Compute the four integer indices.
          ivec2 sourceFloorCR = ivec2(sourceFracIndexCR);
          ivec2 sourceCeilCR = ivec2(ceil(sourceFracIndexCR));

          float topLeft = getImage(b, sourceFloorCR.y, sourceFloorCR.x, d);
          float bottomLeft = getImage(b, sourceCeilCR.y, sourceFloorCR.x, d);
          float topRight = getImage(b, sourceFloorCR.y, sourceCeilCR.x, d);
          float bottomRight = getImage(b, sourceCeilCR.y, sourceCeilCR.x, d);

          vec2 fracCR = sourceFracIndexCR - vec2(sourceFloorCR);

          float top = topLeft + (topRight - topLeft) * fracCR.x;
          float bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR.x;
          float newValue = top + (bottom - top) * fracCR.y;
          setOutput(newValue);
        } else {
          // Compute the coordinators of nearest neighbor point.
          ivec2 sourceNearestCR = ivec2(floor(
            sourceFracIndexCR + vec2(0.5,0.5)));
          float newValue = getImage(b, sourceNearestCR.y, sourceNearestCR.x, d);
          setOutput(newValue);
        }
      }
    `}},mj={kernelName:_t,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{image:i,boxes:a,boxInd:o}=t,{cropSize:s,method:c,extrapolationValue:l}=r,u=new pj(i.shape,a.shape,s,c,l);return n.runWebGLProgram(u,[i,a,o],`float32`)}},hj;(function(e){e.Prod=`*`,e.Sum=`+`})(hj||={});var gj=class{constructor(e,t,n,r){this.op=e,this.outputShape=t,this.variableNames=[`x`],this.customUniforms=[{name:`index`,type:`float`}];let i=this.outputShape.length,a=this.op===hj.Prod?`1.0`:`0.0`,o=n?a:`getX(${_j(i,`coords`,this.op)})`,s=this.outputShape[this.outputShape.length-1],c=``,l=``;n?(c=r?`end != ${s-1}`:`end != 0`,l=r?`end + 1`:`end - 1`):(c=r?`end + pow2 < ${s}`:`end >= pow2`,l=r?`end + pow2`:`end - pow2`),this.userCode=`
      void main() {
        ${Z(i)} coords = getOutputCoords();
        int end = ${vj(i,`coords`,this.op)};
        float val = ${o};
        int pow2 = int(pow(2.0, index));
        if (${c}) {
          int idx = ${l};
          ${vj(i,`coords`,this.op)} = idx;
          val ${this.op}= getX(${_j(i,`coords`,this.op)});
        }
        setOutput(val);
      }
    `}};function _j(e,t,n){if(e===1)return`${t}`;if(e===2)return`${t}.x, ${t}.y`;if(e===3)return`${t}.x, ${t}.y, ${t}.z`;if(e===4)return`${t}.x, ${t}.y, ${t}.z, ${t}.w`;throw Error(`Cumulative ${n} for rank ${e} is not yet supported`)}function vj(e,t,n){if(e===1)return`${t}`;if(e===2)return`${t}.y`;if(e===3)return`${t}.z`;if(e===4)return`${t}.w`;throw Error(`Cumulative ${n} for rank ${e} is not yet supported`)}function yj(e,t,n,r,i,a){let o=t.shape.length,s=Ol([r],o),c=t;s!=null&&(c=tk({inputs:{x:t},backend:n,attrs:{perm:s}}));let l=Al(1,o)[0];if(l!==o-1)throw Error(`WebGL cumprod shader expects an inner-most axis=${t.shape.length-1} but got axis=${r}`);let u=c.shape[l],d=xO({inputs:{x:c},backend:n});for(let t=0;t<=Math.ceil(Math.log2(u))-1;t++){let r=new gj(e,c.shape,!1,a),i=[[t]],o=d;d=n.runWebGLProgram(r,[d],d.dtype,i),n.disposeIntermediateTensorInfo(o)}if(i){let t=new gj(e,c.shape,i,a),r=d;d=n.runWebGLProgram(t,[d],d.dtype),n.disposeIntermediateTensorInfo(r)}if(s!=null){let e=kl(s),t=tk({inputs:{x:d},backend:n,attrs:{perm:e}});return n.disposeIntermediateTensorInfo(d),n.disposeIntermediateTensorInfo(c),t}return d}function bj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,exclusive:o,reverse:s}=r;return yj(hj.Prod,i,n,a,o,s)}var xj={kernelName:ht,backendName:`webgl`,kernelFunc:bj};function Sj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,exclusive:o,reverse:s}=r;return yj(hj.Sum,i,n,a,o,s)}var Cj={kernelName:gt,backendName:`webgl`,kernelFunc:Sj};function wj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,weights:a}=t,{size:o,binaryOutput:s}=r;if(i.shape.length===1){let e=RE(n.readSync(i.dataId),n.readSync(a.dataId),a.dtype,a.shape,o);return n.makeTensorInfo([o],a.dtype,e)}else if(i.shape.length===2){let e=zE(n.bufferSync(i),n.bufferSync(a),o,s);return n.makeTensorInfo(e.shape,a.dtype,e.values)}throw Error(`Error in denseBincount: input must be at most rank 2, but got rank${i.shape.length}.`)}var Tj={kernelName:vt,backendName:`webgl`,kernelFunc:wj},Ej=class{constructor(e,t,n){this.variableNames=[`x`],this.outputShape=[],this.outputShape=e,this.blockSize=t,this.dataFormat=n,this.userCode=`
    void main() {
      ivec4 coords = getOutputCoords();
      int b = coords[0];
      int h = ${this.getHeightCoordString()};
      int w = ${this.getWidthCoordString()};
      int d = ${this.getDepthCoordString()};

      int in_h = h / ${t};
      int offset_h = imod(h, ${t});
      int in_w = w / ${t};
      int offset_w = imod(w, ${t});
      int offset_d = (offset_h * ${t} + offset_w) *
        ${this.getOutputDepthSize()};
      int in_d = d + offset_d;

      float result = ${this.getInputSamplingString()};
      setOutput(result);
    }
  `}getHeightCoordString(){return this.dataFormat===`NHWC`?`coords[1]`:`coords[2]`}getWidthCoordString(){return this.dataFormat===`NHWC`?`coords[2]`:`coords[3]`}getDepthCoordString(){return this.dataFormat===`NHWC`?`coords[3]`:`coords[1]`}getOutputDepthSize(){return this.dataFormat===`NHWC`?this.outputShape[3]:this.outputShape[1]}getInputSamplingString(){return this.dataFormat===`NHWC`?`getX(b, in_h, in_w, in_d)`:`getX(b, in_d, in_h, in_w)`}};function Dj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockSize:a,dataFormat:o}=r,s=i.shape[0],c=o===`NHWC`?i.shape[1]:i.shape[2],l=o===`NHWC`?i.shape[2]:i.shape[3],u=o===`NHWC`?i.shape[3]:i.shape[1],d=c*a,f=l*a,p=u/(a*a),m=new Ej(o===`NHWC`?[s,d,f,p]:[s,p,d,f],a,o);return n.runWebGLProgram(m,[i],i.dtype)}var Oj={kernelName:yt,backendName:`webgl`,kernelFunc:Dj},kj=class{constructor(e,t=!1,n=null,r=!1,i=!1){this.variableNames=[`x`,`W`],this.customUniforms=[{name:`pads`,type:`ivec2`},{name:`strides`,type:`ivec2`},{name:`dilations`,type:`ivec2`},{name:`inDims`,type:`ivec2`}],this.outputShape=e.outShape,this.enableShapeUniforms=Iw(this.outputShape.length);let a=e.filterHeight,o=e.filterWidth,s=e.outChannels/e.inChannels,c=``,l=``;n&&(c=r?`float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${n}
        }`:i?`float activation(float a) {
          float b = getLeakyreluAlphaAtOutCoords();
          ${n}
        }`:`
          float activation(float x) {
            ${n}
          }
        `,l=`result = activation(result);`);let u=t?`result += getBiasAtOutCoords();`:``;t&&this.variableNames.push(`bias`),r&&this.variableNames.push(`preluActivationWeights`),i&&this.variableNames.push(`leakyreluAlpha`),this.userCode=`
      ${c}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${s};
        int q = d2 - d1 * ${s};

        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        // TO DO(dsmilkov): Flatten the two for loops and vec4 the operations.
        for (int wR = 0; wR < ${a}; wR++) {
          int xR = xRCorner + wR * dilations[0];

          if (xR < 0 || xR >= inDims[0]) {
            continue;
          }

          for (int wC = 0; wC < ${o}; wC++) {
            int xC = xCCorner + wC * dilations[1];

            if (xC < 0 || xC >= inDims[1]) {
              continue;
            }

            float xVal = getX(batch, xR, xC, d1);
            float wVal = getW(wR, wC, d1, q);
            dotProd += xVal * wVal;
          }
        }

        float result = dotProd;
        ${u}
        ${l}
        setOutput(result);
      }
    `}},Aj=class{constructor(e,t=!1,n=null,r=!1,i=!1){this.variableNames=[`x`,`W`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`pads`,type:`ivec2`},{name:`strides`,type:`ivec2`},{name:`dilations`,type:`ivec2`},{name:`inDims`,type:`ivec2`}],this.outputShape=e.outShape,this.enableShapeUniforms=Iw(this.outputShape.length);let a=e.outChannels/e.inChannels,o=e.padInfo.left,s=e.strideWidth,c=e.dilationWidth,l=e.filterHeight,u=e.filterWidth,d=u,f=`
      int xR; int xC; int xCOffset;
      vec4 wTexel; vec4 previous; vec4 final;`;for(let e=0;e<u;e++)f+=`
          vec4 xTexelC${e*2};
          int xTexelC${e*2}Ready;
          vec4 xTexelC${e*2+1};
          int xTexelC${e*2+1}Ready;
          vec4 xC${e};`;f+=`
    for (int r = 0; r < ${l}; r++) {
      `;for(let e=0;e<u;e++)f+=`
          xTexelC${e*2} = vec4(0.0);
          xTexelC${e*2}Ready = 0;
          xTexelC${e*2+1} = vec4(0.0);
          xTexelC${e*2+1}Ready = 0;
          xC${e} = vec4(0.0);`;f+=`
        xR = xRCorner + r * dilations[0];
        if (xR >=0 && xR < inDims[0]) {
      `;for(let e=0;e<(d+1)/2;e++){let t=e*2;if(f+=`
          xC = xCCorner + ${t*c};
          `,s===1){if(t<u&&(o%2==1?(f+=`
                xCOffset = xC + 1;
                if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t}Ready == 0) {
                  xTexelC${t} = getX(batch, xR, xCOffset, d1);

                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${t}.zw = vec2(0.0);
                  }
                  xTexelC${t}Ready = 1;
                }
              `,c===1&&t>0?f+=`
                xC${t} = vec4(xTexelC${t-2}.zw, xTexelC${t}.xy);
                `:f+=`
                  xCOffset = xC + 1 - 2;

                  if (xCOffset >= 0 && xCOffset < inDims[1]) {
                    previous = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      previous.zw = vec2(0.0);
                    }

                    xC${t} = vec4(previous.zw, xTexelC${t}.xy);
                  } else {
                    xC${t} = vec4(0.0, 0.0, xTexelC${t}.xy);
                  }
                  `):f+=`
                if (xC >= 0 && xC < inDims[1] && xTexelC${t}Ready == 0) {
                  xTexelC${t} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${t}.zw = vec2(0.0);
                  }
                  xTexelC${t}Ready = 1;
                }

                xC${t} = xTexelC${t};
                `,t+1<u)){let e=o%2==0?g(c):c;c%2==0&&o%2==1||c%2!=0&&o%2!=1?(f+=`
                  xCOffset = xC + imod(pads[1], 2) + ${e};

                  if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t+1}Ready == 0) {
                    xTexelC${t+1} = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      xTexelC${t+1}.zw = vec2(0.0);
                    }
                    xTexelC${t+1}Ready = 1;
                  }
                  `,c>1?f+=`
                    xCOffset -= 2;
                    if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);
                     xC${t+1} = vec4(previous.zw, xTexelC${t+1}.xy);
                    } else {
                     xC${t+1} = vec4(0.0, 0.0, xTexelC${t+1}.xy);
                    }
                    `:f+=`
                    xC${t+1} = vec4(xTexelC${t}.zw, xTexelC${t+1}.xy);
                    `):e===1?f+=`
                    xC${t+1} = xTexelC${t};
                    `:f+=`
                    xCOffset = xC + ${e};

                    if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t+1}Ready == 0) {
                      xTexelC${t+1} = getX(batch, xR, xCOffset, d1);
                      if (xCOffset + 1 >= inDims[1]) {
                        xTexelC${t+1}.zw = vec2(0.0);
                      }
                      xTexelC${t+1}Ready = 1;
                    }

                    xC${t+1} = xTexelC${t+1};
                    `}}else t<u&&(o%2==1?(f+=`
                xCOffset = xC + 1 - strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t}Ready == 0) {
                  xTexelC${t} = getX(batch, xR, xCOffset, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${t}.zw = vec2(0.0);
                  }
                  xTexelC${t}Ready = 1;
                }

                if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${t+1}Ready == 0) {
                  xTexelC${t+1} = getX(batch, xR, xC + 1, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xC + 2 >= inDims[1]) {
                    xTexelC${t+1}.zw = vec2(0.0);
                  }
                  xTexelC${t+1}Ready = 1;
                }

                xC${t} = vec4(xTexelC${t}.zw, xTexelC${t+1}.zw);
              `,t+1<u&&(f+=`
                  final = vec4(0.0);
                  xCOffset = xC + 1 + strides[1];
                  if(xCOffset >= 0 && xCOffset < inDims[1]) {
                    final = getX(batch, xR, xCOffset, d1);
                  }
                  xC${t+1} = vec4(xTexelC${t+1}.xy, final.xy);
                `)):(f+=`
                if(xC >= 0 && xC < inDims[1] && xTexelC${t}Ready == 0) {
                  xTexelC${t} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${t}.zw = vec2(0.0);
                  }
                  xTexelC${t}Ready = 1;
                }

                xCOffset = xC + strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t+1}Ready == 0) {
                  xTexelC${t+1} = getX(batch, xR, xCOffset, d1);
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${t+1}.zw = vec2(0.);
                  }
                  xTexelC${t+1}Ready = 1;
                }

                xC${t} = vec4(
                  xTexelC${t}.xy, xTexelC${t+1}.xy);
              `,t+1<u&&(f+=`
                  xC${t+1} = vec4(xTexelC${t}.zw, xTexelC${t+1}.zw);
                `)));t<u&&(f+=`
            wTexel = getW(r, ${t}, d1, q);
            dotProd += xC${t} * vec4(wTexel.xz, wTexel.xz);
          `,t+1<u&&(f+=`
              wTexel = getW(r, ${t+1}, d1, q);
              dotProd += xC${t+1} * vec4(wTexel.xz, wTexel.xz);
            `))}f+=`
    }
  `,f+=`
      }
    `;let p=``,m=``;n&&(p=r?`vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${n}
        }`:i?`vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${n}
        }`:`vec4 activation(vec4 x) {
          ${n}
        }`,m=`result = activation(result);`);let h=t?`result += getBiasAtOutCoords();`:``;t&&this.variableNames.push(`bias`),r&&this.variableNames.push(`preluActivationWeights`),i&&this.variableNames.push(`leakyreluAlpha`),this.userCode=`
      ${p}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${a};
        int q = d2 - d1 * ${a};
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
        vec4 dotProd = vec4(0.000000000000001);

        ${f}

        vec4 result = dotProd - vec4(0.000000000000001);
        ${h}
        ${m}
        setOutput(result);
      }
    `}};function jj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c,dimRoundingMode:l}=r,u=c;u??=[1,1],y(Os(o,u),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${o} and dilations '${u}'`);let d=gs(i.shape,a.shape,o,u,s,l,!0),f;f=O().getBool(`WEBGL_PACK_DEPTHWISECONV`)&&d.strideWidth<=2&&d.outChannels/d.inChannels===1?new Aj(d):new kj(d);let p=[[d.padInfo.top,d.padInfo.left],[d.strideHeight,d.strideWidth],[d.dilationHeight,d.dilationWidth],[d.inHeight,d.inWidth]];return n.runWebGLProgram(f,[i,a],`float32`,p)}var Mj={kernelName:bt,backendName:`webgl`,kernelFunc:jj},Nj=class{constructor(e){this.variableNames=[`x`,`dy`],this.outputShape=e.filterShape;let t=e.strideHeight,n=e.strideWidth,r=e.padInfo.top,i=e.padInfo.left,a=e.outChannels/e.inChannels;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int dm = coords.w;
        int d2 = d1 * ${a} + dm;

        float dotProd = 0.0;

        // TO DO: Vec4 over the batch size
        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yR = 0; yR < ${e.outHeight}; yR++) {
            int xR = wR + yR * ${t} - ${r};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${e.outWidth}; yC++) {
              int xC = wC + yC * ${n} - ${i};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              float dyValue = getDy(b, yR, yC, d2);
              float xValue = getX(b, xR, xC, d1);
              dotProd += (xValue * dyValue);
            }
          }
        }
        setOutput(dotProd);
      }
    `}},Pj=class{constructor(e){this.variableNames=[`dy`,`W`],this.outputShape=e.inShape;let t=e.filterHeight,n=e.filterWidth,r=e.strideHeight,i=e.strideWidth,a=t-1-e.padInfo.top,o=n-1-e.padInfo.left,s=e.outChannels/e.inChannels;this.userCode=`
      const ivec2 pads = ivec2(${a}, ${o});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[3];
        ivec2 dyCorner = coords.yz - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        float dotProd = 0.0;

        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / ${r}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${n}; wC++) {
            float dyC = float(dyCCorner + wC) / ${i}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${n} - 1 - wC;

            // TO DO: Vec4 over the channelMul
            for (int dm = 0; dm < ${s}; dm++) {
              int d2 = d1 * ${s} + dm;
              float xValue = getDy(batch, idyR, idyC, d2);
              float wValue = getW(wRPerm, wCPerm, d1, dm);
              dotProd += xValue * wValue;
            }
          }
        }
        setOutput(dotProd);
      }
    `}};function Fj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,dilations:s,pad:c,dimRoundingMode:l,filterShape:u}=r,d=new Nj(gs(i.shape,u,o,s,c,l,!0));return n.runWebGLProgram(d,[i,a],`float32`)}var Ij={kernelName:xt,backendName:`webgl`,kernelFunc:Fj};function Lj(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{strides:o,dilations:s,pad:c,dimRoundingMode:l,inputShape:u}=r,d=new Pj(gs(u,a.shape,o,s,c,l,!0));return n.runWebGLProgram(d,[i,a],`float32`)}var Rj={kernelName:St,backendName:`webgl`,kernelFunc:Lj},zj=class{constructor(e){this.variableNames=[`X`],this.outputShape=[e,e],this.userCode=`
      void main() {
          ivec2 coords = getOutputCoords();
          float val = coords[0] == coords[1] ? getX(coords[0]) : 0.0;
          setOutput(val);
      }
    `}};function Bj(e){let{inputs:t,backend:n}=e,{x:r}=t,i=[...r.shape,...r.shape],a=S(r.shape),o=$({inputs:{x:r},backend:n,attrs:{shape:[a]}}),s=new zj(a),c=n.runWebGLProgram(s,[o],o.dtype),l=$({inputs:{x:c},backend:n,attrs:{shape:i}});return n.disposeIntermediateTensorInfo(o),n.disposeIntermediateTensorInfo(c),l}var Vj={kernelName:Ct,backendName:`webgl`,kernelFunc:Bj},Hj=class{constructor(e){this.variableNames=[`x`,`W`],this.outputShape=e.outShape;let{inHeight:t,inWidth:n,padInfo:r,strideHeight:i,strideWidth:a,filterHeight:o,filterWidth:s,dilationHeight:c,dilationWidth:l}=e,{top:u,left:d}=r;this.userCode=`
      const ivec2 strides = ivec2(${i}, ${a});
      const ivec2 pads = ivec2(${u}, ${d});
      const float neg_infinity = -3.4e38;

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.w;
        ivec2 outTopLeftCorner =
            coords.yz * strides - pads;
        int hBeg = outTopLeftCorner.x;
        int wBeg = outTopLeftCorner.y;

        float curVal = neg_infinity;
        for (int h = 0; h < ${o}; h++) {
          int hIn = hBeg + h * ${c};

          if (hIn >= 0 && hIn < ${t}) {
            for (int w = 0; w < ${s}; w++) {
              int wIn = wBeg + w * ${l};

              if (wIn >= 0 && wIn < ${n}) {
                float xVal = getX(batch, hIn, wIn, d1);
                float wVal = getW(h, w, d1);

                float val = xVal + wVal;
                if (val > curVal) {
                  curVal = val;
                }
              }
            }
          }
        }

        float result = curVal;
        setOutput(result);
      }
    `}};function Uj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c}=r,l=ps(i.shape,a.shape,o,s,`NHWC`,c),u,d=new Hj(l);u=n.runWebGLProgram(d,[i,a],`float32`);let f=$({inputs:{x:u},backend:n,attrs:{shape:l.outShape}});return n.disposeIntermediateTensorInfo(u),f}var Wj={kernelName:wt,backendName:`webgl`,kernelFunc:Uj};function Gj(e){let{inputs:t,backend:n,attrs:r}=e,{equation:i}=r,a=t,{allDims:o,summedDims:s,idDims:c}=Kv(i,a.length);Jv(o.length,c,a);let{path:l,steps:u}=Yv(s,c),d=u.length,f=null,p=o.length,m=[];for(let e=0;e<d;++e){for(let t of u[e]){let{permutationIndices:e,expandDims:r}=qv(p,c[t]),i;Xv(e)?i=a[t]:(i=tk({inputs:{x:a[t]},backend:n,attrs:{perm:e}}),m.push(i));let o=i.shape.slice();for(let e=0;e<r.length;++e)o.splice(r[e],0,1);w(i.shape,o)||(i=$({inputs:{x:i},backend:n,attrs:{shape:o}}),m.push(i)),f===null?f=i:(f=BO({inputs:{a:i,b:f},backend:n}),m.push(f))}e<d-1&&(l[e]>=0&&(f=$O({inputs:{x:f},backend:n,attrs:{axis:l[e]-(o.length-p),keepDims:!1}}),m.push(f)),p--)}for(let e of m)e!==f&&n.disposeIntermediateTensorInfo(e);return f}var Kj={kernelName:Et,backendName:`webgl`,kernelFunc:Gj},qj={kernelName:`Elu`,backendName:`webgl`,kernelFunc:Q({opSnippet:`return (x >= 0.0) ? x : (exp(x) - 1.0);`,packedOpSnippet:`
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`})},Jj=`return (b >= 0.0) ? a : a * (b + 1.0);`,Yj=`
  vec4 bGTEZero = vec4(greaterThanEqual(b, vec4(0.)));
  return (bGTEZero * a) + ((vec4(1.0) - bGTEZero) * (a * (b + vec4(1.0))));
`,Xj={kernelName:Dt,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n}=e,{dy:r,y:i}=t,a=O().getBool(`WEBGL_PACK_BINARY_OPERATIONS`)?new bO(Yj,r.shape,i.shape):new vO(Jj,r.shape,i.shape);return n.runWebGLProgram(a,[r,i],r.dtype)}},Zj={kernelName:Ot,backendName:`webgl`,kernelFunc:PO({opSnippet:`return float(a == b);`,packedOpSnippet:`
  return vec4(equal(a, b));
`,dtype:`bool`,cpuKernelImpl:WE})},Qj={kernelName:`Erf`,backendName:`webgl`,kernelFunc:Q({opSnippet:`
  // Error function is calculated approximately with elementary function.
  // See "Handbook of Mathematical Functions with Formulas,
  // Graphs, and Mathematical Tables", Abramowitz and Stegun.
  float p = ${Ov};
  float a1 = ${kv};
  float a2 = ${Av};
  float a3 = ${jv};
  float a4 = ${Mv};
  float a5 = ${Nv};

  float sign = sign(x);
  x = abs(x);
  float t = 1.0 / (1.0 + p * x);
  return sign * (1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x));
`})},$j=Q({opSnippet:NO+`
  return exp(x);
`,packedOpSnippet:`
  vec4 result = exp(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,cpuKernelImpl:GE,dtype:`float32`}),eM={kernelName:`Exp`,backendName:`webgl`,kernelFunc:$j};function tM(e){let{inputs:t,attrs:n,backend:r}=e,{dim:i}=n,{input:a}=t,o=a.shape.length,s=a.shape.slice(),c=i;return i<0&&(y(-(o+1)<=i,()=>`Axis must be in the interval [${-(o+1)}, ${o}]`),c=o+i+1),s.splice(c,0,1),$({inputs:{x:a},backend:r,attrs:{shape:s}})}var nM={kernelName:kt,backendName:`webgl`,kernelFunc:tM},rM=`return exp(x) - 1.0;`,iM={kernelName:At,backendName:`webgl`,kernelFunc:Q({opSnippet:rM,packedOpSnippet:rM,cpuKernelImpl:KE})},aM=class{constructor(e,t,n){this.variableNames=[`real`,`imag`];let r=t[1];this.outputShape=t;let i=n?`2.0 * ${Math.PI}`:`-2.0 * ${Math.PI}`,a=n?`${r}.0`:`1.0`,o;if(e===`real`)o=`return real * expR - imag * expI;`;else if(e===`imag`)o=`return real * expI + imag * expR;`;else throw Error(`FFT component must be either "real" or "imag", got ${e}.`);this.userCode=`
      const float exponentMultiplier = ${i};

      float unaryOpComplex(float real, float expR, float imag, float expI) {
        ${o}
      }

      float mulMatDFT(int batch, int index) {
        float indexRatio = float(index) / float(${r});
        float exponentMultiplierTimesIndexRatio =
            exponentMultiplier * indexRatio;

        float result = 0.0;

        for (int i = 0; i < ${r}; i++) {
          // x = (-2|2 * PI / N) * index * i;
          float x = exponentMultiplierTimesIndexRatio * float(i);
          float expR = cos(x);
          float expI = sin(x);
          float real = getReal(batch, i);
          float imag = getImag(batch, i);

          result +=
              unaryOpComplex(real, expR, imag, expI) / ${a};
        }

        return result;
      }

      void main() {
        ivec2 coords = getOutputCoords();
        setOutput(mulMatDFT(coords[0], coords[1]));
      }
    `}};function oM(e,t,n){let r=n.texData.get(e.dataId),i=S(e.shape),a=e.shape[e.shape.length-1],o=i/a,s=$({inputs:{x:e},backend:n,attrs:{shape:[o,a]}}),c=s.shape,l=new aM(`real`,c,t),u=new aM(`imag`,c,t),d=[{dataId:r.complexTensorInfos.real.dataId,dtype:r.complexTensorInfos.real.dtype,shape:c},{dataId:r.complexTensorInfos.imag.dataId,dtype:r.complexTensorInfos.imag.dtype,shape:c}],f=n.runWebGLProgram(l,d,`float32`),p=n.runWebGLProgram(u,d,`float32`),m=CO({inputs:{real:f,imag:p},backend:n});n.disposeIntermediateTensorInfo(f),n.disposeIntermediateTensorInfo(p);let h=$({inputs:{x:m},backend:n,attrs:{shape:e.shape}});return n.disposeIntermediateTensorInfo(s),n.disposeIntermediateTensorInfo(m),h}function sM(e){let{inputs:t,backend:n}=e,{input:r}=t;return oM(r,!1,n)}var cM={kernelName:`FFT`,backendName:`webgl`,kernelFunc:sM},lM=class{constructor(e,t){this.outputShape=[],this.customUniforms=[{name:`value`,type:`float`}],this.variableNames=[`x`],this.outputShape=e,this.userCode=`
      void main() {
        // Input can be obtained from uniform value.
        setOutput(value);
      }
    `}};function uM(e){let{backend:t,attrs:n}=e,{shape:r,value:i}=n,{dtype:a}=n;if(a||=me(i),a===`string`){let e=ae(a,S(r));return e.fill(i),t.makeTensorInfo(r,a,e)}else{let e=new lM(r,i),n=[[i]];return t.runWebGLProgram(e,[],a,n)}}var dM={kernelName:jt,backendName:`webgl`,kernelFunc:uM},fM=class{constructor(e){this.variableNames=[`Image`],this.outputShape=[];let t=e[2];this.outputShape=e,this.userCode=`
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];

          int coordX = ${t} - x - 1;
          float outputValue;
          if(coordX >= 0 && coordX < ${t}) {
            outputValue = getImage(coords[0], coords[1], coordX, coords[3]);
          } else {
            outputValue = getImage(coords[0], coords[1], coords[2], coords[3]);
          }
          setOutput(outputValue);
        }
    `}},pM={kernelName:Mt,backendName:`webgl`,kernelFunc:({inputs:e,backend:t})=>{let{image:n}=e,r=t,i=new fM(n.shape);return r.runWebGLProgram(i,[n],n.dtype)}},mM=`return floor(x);`,hM={kernelName:Nt,backendName:`webgl`,kernelFunc:Q({opSnippet:mM,packedOpSnippet:mM,cpuKernelImpl:qE})},gM={kernelName:Pt,backendName:`webgl`,kernelFunc:PO({opSnippet:`
  float s = sign(a) * sign(b);
  int ia = round(a);
  int ib = round(b);
  if (ib != 0) {
    // Windows (D3D) wants guaranteed non-zero int division at compile-time.
    return float(idiv(ia, ib, s));
  } else {
    return NAN;
  }
`,packedOpSnippet:`
  ivec4 ia = round(a);
  ivec4 ib = round(b);
  bvec4 cond = notEqual(ib, ivec4(0));
  ivec4 result = ivec4(0);
  vec4 s = sign(a) * sign(b);

  // Windows (D3D) wants guaranteed non-zero int division at compile-time.
  if (cond[0]) {
    result[0] = idiv(ia[0], ib[0], s[0]);
  }
  if (cond[1]) {
    result[1] = idiv(ia[1], ib[1], s[1]);
  }
  if (cond[2]) {
    result[2] = idiv(ia[2], ib[2], s[2]);
  }
  if (cond[3]) {
    result[3] = idiv(ia[3], ib[3], s[3]);
  }
  return vec4(result);
`,dtype:`int32`})},_M=class{constructor(e){this.variableNames=[`A`];let t=MC(),[n,r]=e;this.outputShape=e,this.userCode=`
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${r}.0, ${n}.0);

        vec4 values = ${t.texture2D}(A, uv);
        float value;
        if (depth == 0) {
          value = values.r;
        } else if (depth == 1) {
          value = values.g;
        } else if (depth == 2) {
          value = values.b;
        } else if (depth == 3) {
          value = values.a;
        }

        setOutput(floor(value * 255.0 + 0.5));
      }
    `}},vM=class{constructor(e){this.variableNames=[`A`],this.packedInputs=!1,this.packedOutput=!0;let t=MC(),[n,r]=e;this.outputShape=e,this.userCode=`
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];

        vec4 result = vec4(0.);

        for(int row=0; row<=1; row++) {
          for(int col=0; col<=1; col++) {
            texC = coords[1] + row;
            depth = coords[2] + col;

            vec2 uv = (vec2(texC, texR) + halfCR) /
                       vec2(${r}.0, ${n}.0);
            vec4 values = ${t.texture2D}(A, uv);
            float value;
            if (depth == 0) {
              value = values.r;
            } else if (depth == 1) {
              value = values.g;
            } else if (depth == 2) {
              value = values.b;
            } else if (depth == 3) {
              value = values.a;
            }

            result[row * 2 + col] = floor(value * 255.0 + 0.5);
          }
        }

        ${t.output} = result;
      }
    `}},yM={kernelName:xr,backendName:`webgl`,kernelFunc:SM},bM,xM=O().getBool(`CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU`);function SM(e){let{inputs:t,backend:n,attrs:r}=e,{pixels:i}=t,{numChannels:a}=r,o=typeof HTMLVideoElement<`u`&&i instanceof HTMLVideoElement,s=typeof HTMLImageElement<`u`&&i instanceof HTMLImageElement,[c,l]=o?[i.videoWidth,i.videoHeight]:[i.width,i.height],u=[l,c],d=[l,c,a];if(s||o){let e=O().getBool(`CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU`);(bM==null||e!==xM)&&(xM=e,bM=document.createElement(`canvas`).getContext(`2d`,{willReadFrequently:xM})),bM.canvas.width=c,bM.canvas.height=l,bM.drawImage(i,0,0,c,l),i=bM.canvas}let f=n.makeTensorInfo(u,`int32`);n.texData.get(f.dataId).usage=AS.PIXELS,n.gpgpu.uploadPixelDataToTexture(n.getTexture(f.dataId),i);let p=O().getBool(`WEBGL_PACK`)?new vM(d):new _M(d),m=n.runWebGLProgram(p,[f],`int32`);return n.disposeData(f.dataId),m}function CM(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,bias:o,preluActivationWeights:s}=t,{strides:c,pad:l,dataFormat:u,dilations:d,dimRoundingMode:f,activation:p,leakyreluAlpha:m}=r,h=As(u),g=gs(i.shape,a.shape,c,d,l,f,!1,h),_,v=[],y=o!=null,b=s!=null,x=p===`leakyrelu`,S=()=>{let e=[i,a],t=(e,t)=>{if(t===`NCHW`&&e.shape.length===1&&e.shape[0]!==1){let t=$({inputs:{x:e},backend:n,attrs:{shape:[e.shape[0],1,1]}});return v.push(t),t}return e};if(y&&e.push(t(o,u)),b&&e.push(t(s,u)),x){let t=n.makeTensorInfo([],`float32`,$r(m,`float32`));e.push(t),v.push(t)}return e};if(g.filterHeight===1&&g.filterWidth===1&&g.dilationHeight===1&&g.dilationWidth===1&&g.strideHeight===1&&g.strideWidth===1&&(g.padInfo.type===`SAME`||g.padInfo.type===`VALID`))_=KA({x:i,filter:a,convInfo:g,backend:n,bias:o,activation:p,preluActivationWeights:s,leakyreluAlpha:m});else if(g.strideWidth<=2&&h===`channelsLast`&&O().getBool(`WEBGL_EXP_CONV`)){let e=new UA(g,y,p?FO(p,!0):null,b,x),t=[[g.padInfo.top,g.padInfo.left],[g.strideHeight,g.strideWidth],[g.dilationHeight,g.dilationWidth],[g.inHeight,g.inWidth]],r=S();_=n.runWebGLProgram(e,r,`float32`,t)}else if(O().getBool(`WEBGL_CONV_IM2COL`))_=qA({x:i,filter:a,convInfo:g,backend:n,bias:o,activation:p,preluActivationWeights:s,leakyreluAlpha:m});else{let e=new VA(g,y,p?FO(p,!1):null,b,x),t=S();_=n.runWebGLProgram(e,t,`float32`)}let C=$({inputs:{x:_},backend:n,attrs:{shape:g.outShape}});return v.push(_),v.forEach(e=>n.disposeIntermediateTensorInfo(e)),C}var wM={kernelName:wr,backendName:`webgl`,kernelFunc:CM};function TM(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,bias:o,preluActivationWeights:s}=t,{strides:c,pad:l,dilations:u,dimRoundingMode:d,activation:f,leakyreluAlpha:p}=r,m=[],h=u;h??=[1,1],y(Os(c,h),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${c} and dilations '${h}'`);let g=gs(i.shape,a.shape,c,h,l,d,!0),_=O().getBool(`WEBGL_PACK_DEPTHWISECONV`)&&g.strideWidth<=2&&g.outChannels/g.inChannels===1,v=f?FO(f,_):null,b=[i,a],x=o!=null,S=s!=null,C=f===`leakyrelu`;if(x&&b.push(o),S&&b.push(s),C){let e=n.makeTensorInfo([],`float32`,$r(p,`float32`));b.push(e),m.push(e)}let w;w=_?new Aj(g,x,v,S,C):new kj(g,x,v,S,C);let ee=[[g.padInfo.top,g.padInfo.left],[g.strideHeight,g.strideWidth],[g.dilationHeight,g.dilationWidth],[g.inHeight,g.inWidth]],T=n.runWebGLProgram(w,b,`float32`,ee);return m.forEach(e=>n.disposeIntermediateTensorInfo(e)),T}var EM={kernelName:Tr,backendName:`webgl`,kernelFunc:TM},DM=class{constructor(e,t,n,r){this.sliceDim=e,this.strides=t,this.paramsShape=r,this.variableNames=[`x`,`indices`],this.outputShape=n;let i=Z(n.length),a=`
    int index;`;for(let e=0;e<this.sliceDim;e++)a+=`
          index = round(getIndices(coords[0], ${e}));
          out_of_bounds = out_of_bounds || index < 0;
          out_of_bounds = out_of_bounds || index >= ${this.paramsShape[e]};
          flattenIndex += index * ${this.strides[e]};`;this.userCode=`
         void main() {
          ${i} coords = getOutputCoords();
          int flattenIndex = 0;
          bool out_of_bounds = false;

          ${a}

          setOutput(out_of_bounds ? 0.0 : getX(flattenIndex, coords[1]));
        }
      `}};function OM(e){let{inputs:t,backend:n}=e,{params:r,indices:i}=t,a=i.shape,o=a[a.length-1],s=S(r.shape),[c,l,u,d]=H_(r,i),f=$({inputs:{x:i},backend:n,attrs:{shape:[l,o]}}),p=$({inputs:{x:r},backend:n,attrs:{shape:[S(r.shape)/u,u]}});if(n.shouldExecuteOnCPU([r,i])||r.dtype===`string`){let e=JE(n.readSync(i.dataId),n.bufferSync(r),r.dtype,l,o,u,d,r.shape,s);return n.makeTensorInfo(c,r.dtype,e.values)}let m=new DM(o,d,[l,u],r.shape),h=n.runWebGLProgram(m,[p,f],p.dtype),g=$({inputs:{x:h},backend:n,attrs:{shape:c}});return n.disposeIntermediateTensorInfo(f),n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(h),g}var kM={kernelName:Lt,backendName:`webgl`,kernelFunc:OM},AM=class{constructor(e,t){this.variableNames=[`A`,`indices`],this.outputShape=t,this.rank=t.length;let n=Z(this.rank),r=jM(e,2);this.userCode=`
      void main() {
        ${n} resRC = getOutputCoords();
        int index = int(getIndices(resRC.x, resRC.z));
        float inBounds = (index >= 0) && (index < ${e[2]}) ? 1.0 : 0.0;
        setOutput(inBounds * getA(${r}));
      }
    `}};function jM(e,t){let n=[`resRC.x`,`resRC.y`,`resRC.z`,`resRC.w`],r=[];for(let t=0;t<e.length;t++)t===2?r.push(`index`):r.push(`${n[t]}`);return r.join()}function MM(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,indices:a}=t,{axis:o,batchDims:s}=r,c=D(o,i.shape)[0];if(O().get(`DEBUG`)){let e=n.readSync(a.dataId),t=i.shape[c];for(let n=0;n<e.length;++n){let r=e[n];y(r<=t-1&&r>=0,()=>`GatherV2: the index value ${r} is not in [0, ${t-1}]`)}}let l=my(i,a,c,s),u=S(a.shape),d=[],f=$({inputs:{x:i},backend:n,attrs:{shape:[l.batchSize,l.outerSize,l.dimSize,l.sliceSize]}}),p=$({inputs:{x:a},backend:n,attrs:{shape:[l.batchSize,u/l.batchSize]}});d.push(f),d.push(p);let m=[l.batchSize,l.outerSize,u/l.batchSize,l.sliceSize];if(n.shouldExecuteOnCPU([i,a])||i.dtype===`string`){let e=n.bufferSync(p),t=YE(n.bufferSync(f),e,m);return d.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.makeTensorInfo(l.outputShape,t.dtype,t.values)}let h=new AM(f.shape,m),g=n.runWebGLProgram(h,[f,p],f.dtype);d.push(g);let _=$({inputs:{x:g},backend:n,attrs:{shape:l.outputShape}});return d.forEach(e=>n.disposeIntermediateTensorInfo(e)),_}var NM={kernelName:It,backendName:`webgl`,kernelFunc:MM},PM={kernelName:Rt,backendName:`webgl`,kernelFunc:PO({opSnippet:`return float(a > b);`,packedOpSnippet:`
  return vec4(greaterThan(a, b));
`,cpuKernelImpl:XE,dtype:`bool`})},FM={kernelName:zt,backendName:`webgl`,kernelFunc:PO({opSnippet:`return float(a >= b);`,packedOpSnippet:`
  return vec4(greaterThanEqual(a, b));
`,dtype:`bool`,cpuKernelImpl:ZE})};function IM(e){let{inputs:t,backend:n}=e,{input:r}=t;return oM(r,!0,n)}var LM={kernelName:Vt,backendName:`webgl`,kernelFunc:IM},RM={kernelName:Ut,backendName:`webgl`,kernelFunc:Q({opSnippet:`return float(!isnan(x) && !isinf(x));`,dtype:`bool`})},zM={kernelName:Wt,backendName:`webgl`,kernelFunc:Q({opSnippet:`return float(isinf(x));`,dtype:`bool`})},BM={kernelName:Gt,backendName:`webgl`,kernelFunc:Q({opSnippet:`return float(isnan(x));`,dtype:`bool`})},VM={kernelName:qt,backendName:`webgl`,kernelFunc:PO({opSnippet:`return float(a < b);`,packedOpSnippet:`
  return vec4(lessThan(a, b));
`,cpuKernelImpl:QE,dtype:`bool`})},HM={kernelName:Jt,backendName:`webgl`,kernelFunc:PO({opSnippet:`return float(a <= b);`,packedOpSnippet:`
  return vec4(lessThanEqual(a, b));
`,cpuKernelImpl:$E,dtype:`bool`})};function UM(e){let{backend:t,attrs:n}=e,{start:r,stop:i,num:a}=n,o=eD(r,i,a);return t.makeTensorInfo([o.length],`float32`,o)}var WM={kernelName:Yt,backendName:`webgl`,kernelFunc:UM},GM={kernelName:`Log`,backendName:`webgl`,kernelFunc:Q({opSnippet:NO+`
  return x < 0.0 ? 0./0. : log(x);
`,packedOpSnippet:`
  vec4 result = log(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : (x.r < 0.0 ? 0./0. : result.r);
  result.g = isNaN.g ? x.g : (x.g < 0.0 ? 0./0. : result.g);
  result.b = isNaN.b ? x.b : (x.b < 0.0 ? 0./0. : result.b);
  result.a = isNaN.a ? x.a : (x.a < 0.0 ? 0./0. : result.a);
  return result;
`,cpuKernelImpl:tD})},KM={kernelName:Xt,backendName:`webgl`,kernelFunc:Q({opSnippet:NO+`
  return log(1.0 + x);
`})},qM={kernelName:Zt,backendName:`webgl`,kernelFunc:PO({opSnippet:`return float(a >= 1.0 && b >= 1.0);`,packedOpSnippet:`
  return vec4(
    vec4(greaterThanEqual(a, vec4(1.0))) *
    vec4(greaterThanEqual(b, vec4(1.0))));
`,dtype:`bool`})},JM={kernelName:Qt,backendName:`webgl`,kernelFunc:Q({opSnippet:`return float(!(x >= 1.0));`})},YM={kernelName:$t,backendName:`webgl`,kernelFunc:PO({opSnippet:`return float(a >= 1.0 || b >= 1.0);`,packedOpSnippet:`
  return min(
    vec4(greaterThanEqual(a, vec4(1.0))) +
    vec4(greaterThanEqual(b, vec4(1.0))),
    vec4(1.0));
`,dtype:`bool`})},XM=class{constructor(e,t,n,r,i){this.variableNames=[`x`],this.outputShape=[];let a=t,o=e[3]-1;this.outputShape=e;let s,c=`float(${n}) + float(${r}) * sum`;s=i===.5?`inversesqrt(${c})`:i===1?`1.0/(${c})`:`exp(log(${c}) * float(-${i}));`,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];
        int d = coords[3];
        float x = getX(b, r, c, d);
        float sum = 0.0;
        for (int j = -${a}; j <= ${a}; j++) {
          int idx = d + j;
          if (idx >= 0 && idx <=  ${o}) {
            float z = getX(b, r, c, idx);
            sum += z * z;
          }
        }
        float val = x * ${s};
        setOutput(val);
      }
    `}},ZM=class{constructor(e,t,n,r,i){this.variableNames=[`x`],this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0;let a=t,o=e[3]-1;this.outputShape=e;let s,c=`float(${n}) + float(${r}) * sum`;s=i===.5?`inversesqrt(${c})`:i===1?`1.0/(${c})`:`exp(log(${c}) * float(-${i}));`,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords.x;
        int r = coords.y;
        int c = coords.z;
        int d = coords.w;

        bool hasNextCol = d < ${this.outputShape[3]};
        bool hasNextRow = c < ${this.outputShape[2]};

        vec4 sum = vec4(0.);
        vec4 xFragAtOutputCoords = getX(b, r, c, d);

        vec4 xAtOutputCoords = vec4(
          getChannel(xFragAtOutputCoords, vec2(c, d)),
          hasNextCol ?
            getChannel(xFragAtOutputCoords, vec2(c, d + 1)) : 0.0,
          hasNextRow ?
            getChannel(xFragAtOutputCoords , vec2(c + 1, d)) : 0.0,
          (hasNextRow && hasNextCol) ?
            getChannel(xFragAtOutputCoords, vec2(c + 1, d + 1)) : 0.0
        );

        int firstChannel = d - ${a};
        vec2 cache = vec2(0.);
        if(firstChannel >= 0){
          vec4 firstChannelFrag = getX(b, r, c, firstChannel);
          cache.x = getChannel(firstChannelFrag, vec2(c, firstChannel));
            if(hasNextRow){
              cache.y = getChannel(firstChannelFrag, vec2(c + 1, firstChannel));
            }
        }

        ivec2 depth = ivec2(d, d + 1);
        for (int j = - ${a}; j <= ${a}; j++) {
          ivec2 idx = depth + j;
          bvec2 aboveLowerBound = greaterThanEqual(idx, ivec2(0));
          bvec2 belowUpperBound = lessThanEqual(idx, ivec2(${o}));

          bool depthInRange = aboveLowerBound.x && belowUpperBound.x;
          bool depthPlusOneInRange = aboveLowerBound.y && belowUpperBound.y;

          if(depthInRange || depthPlusOneInRange){
            vec4 z = vec4(0.);
            vec4 xFragAtCurrentDepth;
            z.xz = cache.xy;
            if(depthPlusOneInRange && hasNextCol){
              xFragAtCurrentDepth = idx.y != d ?
                getX(b, r, c, idx.y) : xFragAtOutputCoords;
              z.y = getChannel(xFragAtCurrentDepth, vec2(c, idx.y));
              if(hasNextRow){
                z.w = getChannel(xFragAtCurrentDepth, vec2(c + 1, idx.y));
              }
            }
            cache.xy = z.yw;
            sum += z * z;
          }
        }
        vec4 result = xAtOutputCoords * ${s};
        setOutput(result);
      }
    `}},QM={kernelName:`LRN`,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{depthRadius:a,bias:o,alpha:s,beta:c}=r,l=O().getBool(`WEBGL_PACK_NORMALIZATION`)?new ZM(i.shape,a,o,s,c):new XM(i.shape,a,o,s,c);return n.runWebGLProgram(l,[i],i.dtype)}},$M=class{constructor(e,t,n,r,i){this.variableNames=[`inputImage`,`outputImage`,`dy`],this.outputShape=[],this.outputShape=e,this.depth=e[3],this.depthRadius=t,this.bias=n,this.alpha=r,this.beta=i,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];

        float result = 0.0;
        for (int d = 0; d < ${this.depth}; ++d) {
          int depthBegin = int(max(0.0, float(d - ${t})));
          int depthEnd = int(min(float(${this.depth}),
              float(d + ${t} + 1)));

          const int MIN_DEPTH_BEGIN = 0;
          const int MAX_DEPTH_END = ${this.depth};

          float norm = 0.0;
          for (int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k) {
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd) {
              norm += getInputImage(b, r, c, k) * getInputImage(b, r, c, k);
            }
            else {
              break;
            }
          }

          norm = float(${r}) * norm + float(${n});

          for(int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k){
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd){
              float dyi = -2.0 * float(${r})
                * float(${i})
                * getInputImage(b, r, c, k) * getOutputImage(b, r, c, d)
                / norm;
              if (k == d) {
                dyi += pow(norm, -1.0 * ${i});
              }
              if (k == coords[3]) {
                dyi *= getDy(b, r, c, d);
                result += dyi;
              }
            }
            else {
              break;
            }
          }
      }
      setOutput(result);
      }
    `}},eN={kernelName:en,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{x:i,y:a,dy:o}=t,{depthRadius:s,bias:c,alpha:l,beta:u}=r,d=new $M(i.shape,s,c,l,u);return n.runWebGLProgram(d,[i,a,o],i.dtype)}};function tN(e,t,n,r){let i=S(t),a=S(e.shape)/i,o=$({inputs:{x:e},attrs:{shape:[a,i]},backend:r}),s=qO(o,e.dtype,`max`,r),c=$({inputs:{x:s},attrs:{shape:n},backend:r});return r.disposeIntermediateTensorInfo(o),r.disposeIntermediateTensorInfo(s),c}function nN(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{reductionIndices:a,keepDims:o}=r,s=i.shape.length,c=D(a,i.shape),l=c,u=Ol(l,s),d=u!=null,f=n.shouldExecuteOnCPU([i]),p=i;if(d){if(f){let e=n.texData.get(p.dataId).values,t=Array(s);for(let e=0;e<t.length;e++)t[e]=i.shape[u[e]];let r=AD(e,i.shape,i.dtype,u,t);p=n.makeTensorInfo(t,i.dtype);let a=n.texData.get(p.dataId);a.values=r}else p=ZO(i,u,n);l=Al(l.length,s)}Dl(`max`,l,s);let[m,h]=Tl(p.shape,l),g=m;o&&(g=El(m,c));let _;if(f){let e=n.texData.get(p.dataId).values,t=nD(e,S(h),g,i.dtype);_=n.makeTensorInfo(g,i.dtype);let r=n.texData.get(_.dataId);r.values=t}else _=tN(p,h,g,n);return d&&n.disposeIntermediateTensorInfo(p),_}var rN={kernelName:`Max`,backendName:`webgl`,kernelFunc:nN},iN={kernelName:tn,backendName:`webgl`,kernelFunc:PO({opSnippet:_O+`
  return max(a, b);
`,packedOpSnippet:`
  vec4 result = vec4(max(a, b));
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+yO+`
  return result;
`,cpuKernelImpl:rD})};function aN(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t;jC(i,`maxPool`);let{filterSize:a,strides:o,pad:s,dimRoundingMode:c}=r;y(Os(o,1),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${o} and dilations '1'`);let l=ms(i.shape,a,o,1,s,c);if(l.filterWidth===1&&l.filterHeight===1&&w(l.inShape,l.outShape))return xO({inputs:{x:i},backend:n});let u=new Fk(l,`max`,!1);return n.runWebGLProgram(u,[i],i.dtype)}var oN={kernelName:nn,backendName:`webgl`,kernelFunc:aN};function sN(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dataFormat:c,dimRoundingMode:l}=r,u=new Ik(hs(i.shape,a,o,[1,1,1],s,l,c),`max`,!1);return n.runWebGLProgram(u,[i],i.dtype)}var cN={kernelName:an,backendName:`webgl`,kernelFunc:sN},lN=class{constructor(e){this.variableNames=[`dy`,`maxPos`],this.outputShape=e.inShape;let t=e.strideHeight,n=e.strideWidth,r=e.dilationHeight,i=e.effectiveFilterHeight,a=e.effectiveFilterWidth,o=i-1-e.padInfo.top,s=a-1-e.padInfo.left,c=i*a-1;this.userCode=`
      const ivec2 pads = ivec2(${o}, ${s});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${i};
          wR += ${r}) {
          float dyR = float(dyRCorner + wR) / ${t}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${a}; wC++) {
            float dyC = float(dyCCorner + wC) / ${n}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);
            int maxPosValue = ${c} - int(getMaxPos(b, idyR, idyC, d));

            // Get the current value, check it against the value from the
            // position matrix.
            int curPosValue = wR * ${a} + wC;
            float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

            dotProd += dyValue * mask;
          }
        }
        setOutput(dotProd);
      }
    `}},uN=class{constructor(e){this.variableNames=[`dy`,`maxPos`],this.outputShape=e.inShape;let t=e.strideDepth,n=e.strideHeight,r=e.strideWidth,i=e.dilationDepth,a=e.dilationHeight,o=e.dilationWidth,s=e.effectiveFilterDepth,c=e.effectiveFilterHeight,l=e.effectiveFilterWidth,u=s-1-e.padInfo.front,d=c-1-e.padInfo.top,f=l-1-e.padInfo.left,p=s*c*l-1;this.userCode=`
      const ivec3 pads = ivec3(${u}, ${d}, ${f});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, ch) with pos mask(:, :, :, d) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${s};
           wD += ${i}) {
          float dyD = float(dyDCorner + wD) / ${t}.0;

          if (dyD < 0.0 || dyD >= ${e.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${c};
              wR += ${a}) {
            float dyR = float(dyRCorner + wR) / ${n}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${l};
                wC += ${o}) {
              float dyC = float(dyCCorner + wC) / ${r}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);
              int maxPosValue = ${p} -
                  int(getMaxPos(batch, idyD, idyR, idyC, ch));

              // Get the current value, check it against the value from the
              // position matrix.
              int curPosValue =
                  wD * ${c} * ${l} +
                  wR * ${l} + wC;
              float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

              dotProd += dyValue * mask;
            }
          }
        }
        setOutput(dotProd);
      }
    `}};function dN(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,o=a,{filterSize:s,strides:c,pad:l,dimRoundingMode:u}=r,d=hs(o.shape,s,c,[1,1,1],l,u),f=new Ik(d,`max`,!0),p=n.runWebGLProgram(f,[o],o.dtype),m=new uN(d),h=n.runWebGLProgram(m,[i,p],o.dtype);return n.disposeIntermediateTensorInfo(p),h}var fN={kernelName:on,backendName:`webgl`,kernelFunc:dN};function pN(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a,output:o}=t,s=a;jC([a,o],`maxPoolGrad`);let{filterSize:c,strides:l,pad:u,dimRoundingMode:d}=r,f=ms(s.shape,c,l,1,u,d),p=new Fk(f,`max`,!0),m=n.runWebGLProgram(p,[s],s.dtype),h=new lN(f),g=n.runWebGLProgram(h,[i,m],s.dtype);return n.disposeIntermediateTensorInfo(m),g}var mN={kernelName:rn,backendName:`webgl`,kernelFunc:pN};function hN(e,t,n,r){let i=new Fk(n,`max`,!1),a=r.runWebGLProgram(i,[e],`float32`);return i=new Fk(n,`max`,!0,!0,t),[a,r.runWebGLProgram(i,[e],`float32`)]}var gN={kernelName:sn,backendName:`webgl`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{x:r}=e,{filterSize:i,strides:a,pad:o,includeBatchInIndex:s}=t,c=n;y(r.shape.length===4,()=>`Error in maxPool: input must be rank 4 but got rank ${r.shape.length}.`);let l=[1,1];y(Os(a,l),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${a} and dilations '${l}'`);let[u,d]=hN(r,s,ms(r.shape,i,a,l,o),c);return[u,d]}};function _N(e,t,n,r){let i=S(t),a=S(e.shape)/i,o=$({inputs:{x:e},attrs:{shape:[a,i]},backend:r}),s=qO(o,`float32`,`mean`,r),c=$({inputs:{x:s},attrs:{shape:n},backend:r});return r.disposeIntermediateTensorInfo(o),r.disposeIntermediateTensorInfo(s),c}var vN={kernelName:cn,backendName:`webgl`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{x:r}=e,{keepDims:i,axis:a}=t,o=n,s=r.shape.length,c=D(a,r.shape),l=c,u=Ol(l,s),d=u!=null,f=o.shouldExecuteOnCPU([r]),p=[],m=r;if(d){if(f){let e=o.texData.get(m.dataId).values,t=Array(s);for(let e=0;e<t.length;e++)t[e]=r.shape[u[e]];let n=AD(e,r.shape,r.dtype,u,t);m=o.makeTensorInfo(t,r.dtype);let i=o.texData.get(m.dataId);i.values=n}else m=ZO(r,u,o);p.push(m),l=Al(l.length,s)}Dl(`sum`,l,s);let[h,g]=Tl(m.shape,l),_=h;i&&(_=El(h,c));let v=_N(m,g,_,o);for(let e of p)o.disposeIntermediateTensorInfo(e);return v}};function yN(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r,s=i.shape.length,c=D(a,i.shape),l=c,u=Ol(l,s),d=i;u!=null&&(d=tk({inputs:{x:i},backend:n,attrs:{perm:u}}),l=Al(l.length,i.shape.length)),Dl(`min`,l,s);let[f,p]=Tl(d.shape,l),m=S(p),h=$({inputs:{x:d},backend:n,attrs:{shape:[-1,m]}}),g=qO(h,h.dtype,`min`,n),_;if(o){let e=El(f,c);_=$({inputs:{x:g},backend:n,attrs:{shape:e}})}else _=$({inputs:{x:g},backend:n,attrs:{shape:f}});return n.disposeIntermediateTensorInfo(h),n.disposeIntermediateTensorInfo(g),u!=null&&n.disposeIntermediateTensorInfo(d),_}var bN={kernelName:`Min`,backendName:`webgl`,kernelFunc:yN},xN={kernelName:ln,backendName:`webgl`,kernelFunc:PO({opSnippet:_O+`
  return min(a, b);
`,packedOpSnippet:`
  vec4 result = vec4(min(a, b));
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+yO+`
  return result;
`,cpuKernelImpl:iD})},SN=class{constructor(e,t,n){this.variableNames=[`x`],this.outputShape=t.map((t,n)=>t[0]+e[n]+t[1]);let r=e.length,i=Z(r),a=t.map(e=>e[0]).join(`,`),o=t.map((t,n)=>t[0]+e[n]).join(`,`),s=[`coords[0]`,`coords[1]`,`coords[2]`,`coords[3]`].slice(0,r),c=n===`reflect`?0:1;if(r===1){this.userCode=`
        int start = ${a};
        int end = ${o};

        void main() {
          int outC = getOutputCoords();
          if (outC < start) {
            outC = start * 2 - outC - ${c};
          } else if(outC >= end) {
            outC = (end - 1) * 2 - outC + ${c};
          }
          setOutput(getX(outC - start));
        }
      `;return}this.userCode=`
      ${i} start = ${i}(${a});
      ${i} end = ${i}(${o});

      void main() {
        ${i} outC = getOutputCoords();
        for (int i = 0; i < ${r}; i++) {
          if (outC[i] < start[i]) {
            outC[i] = start[i] * 2 - outC[i] - ${c};
          } else if(outC[i] >= end[i]) {
            outC[i] = (end[i] - 1) * 2 - outC[i] + ${c};
          }
        }
        ${i} coords = outC - start;
        setOutput(getX(${s}));
      }
    `}},CN=class{constructor(e,t,n){this.variableNames=[`x`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t.map((t,n)=>t[0]+e[n]+t[1]);let r=e.length,i=Z(r),a=t.map(e=>e[0]).join(`,`),o=t.map((t,n)=>t[0]+e[n]).join(`,`),s=ND(`rc`,r),c=ND(`source`,r),l=`${s[r-1]} < ${this.outputShape[r-1]}`,u=r===1?`source`:`vec2(${c.slice(-2).join()})`,d=n===`reflect`?0:1,f=``;if(r===1){let e=`
        ${i} source = rc;
        if (source < start) {
          source = start * 2 - source - ${d};
        } else if (source >= end) {
          source = (end - 1) * 2 - source + ${d};
        }
        source -= start;
      `;f=`
        ${i} rc = outputLoc;
        ${e}
        result[0] = getChannel(getX(${c.join()}), ${u});
        ${s[r-1]} += 1;
        if(${l}) {
          ${e}
          result[1] = getChannel(getX(${c.join()}), ${u});
        }
      `}else{let e=`
        ${i} source = rc;
        ${i} lt = ${i}(lessThan(source, start));
        ${i} gte = ${i}(greaterThanEqual(source, end));
        ${i} orig = 1 - (lt + gte);
        source = orig * source +
                lt * (start * 2 - source - ${d}) +
                gte * ((end - 1) * 2 - source + ${d});
        source -= start;
      `;f=`
        ${i} rc = outputLoc;
        ${e}
        result[0] = getChannel(getX(${c.join()}), ${u});
        ${s[r-1]} += 1;
        if(${l}) {
          ${e}
          result[1] = getChannel(getX(${c.join()}), ${u});
        }
        rc = outputLoc;
        ${s[r-2]} += 1;
        if(${s[r-2]} < ${this.outputShape[r-2]}) {
          ${e}
          result[2] = getChannel(getX(${c.join()}), ${u});
          ${s[r-1]} += 1;
          if(${l}) {
            ${e}
            result[3] = getChannel(getX(${c.join()}), ${u});
          }
        }
      `}this.userCode=`
      const ${i} start = ${i}(${a});
      const ${i} end = ${i}(${o});

      void main() {
        ${i} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${f}
        setOutput(result);
      }
    `}},wN={kernelName:un,backendName:`webgl`,kernelFunc:({inputs:e,backend:t,attrs:n})=>{let{x:r}=e,{paddings:i,mode:a}=n,o=O().getBool(`WEBGL_PACK_ARRAY_OPERATIONS`)?new CN(r.shape,i,a):new SN(r.shape,i,a);return t.runWebGLProgram(o,[r],r.dtype)}},TN={kernelName:`Mod`,backendName:`webgl`,kernelFunc:PO({opSnippet:`if (b == 0.0) return NAN;
  return mod(a, b);`,packedOpSnippet:`
  vec4 result = mod(a, b);
  bvec4 isNaN = equal(b, vec4(0.0));
  `+yO+`
  return result;
`})},EN=class{constructor(e,t,n){this.variableNames=[`probs`],this.customUniforms=[{name:`seed`,type:`float`}],this.outputShape=[e,n],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];

        float r = random(seed);
        float cdf = 0.0;

        for (int i = 0; i < ${t-1}; i++) {
          cdf += getProbs(batch, i);

          if (r < cdf) {
            setOutput(float(i));
            return;
          }
        }

        // If no other event happened, last event happened.
        setOutput(float(${t-1}));
      }
    `}},DN=PO({opSnippet:`
if (a == b) {
  return 1.0;
};
return a / b;`,packedOpSnippet:`
  // vec4 one = vec4(equal(a, b));
  // return one + (vec4(1.0) - one) * a / b;
  vec4 result = a / b;
  if(a.x == b.x) {
    result.x = 1.;
  }
  if(a.y == b.y) {
    result.y = 1.;
  }
  if(a.z == b.z) {
    result.z = 1.;
  }
  if(a.w == b.w) {
    result.w = 1.;
  }

  return result;
`,checkOutOfBounds:!0}),ON={kernelName:Tt,backendName:`webgl`,kernelFunc:DN},kN=`return a - b;`,AN=PO({opSnippet:kN,packedOpSnippet:kN,supportsComplex:!0,cpuKernelImpl:DD}),jN={kernelName:`Sub`,backendName:`webgl`,kernelFunc:AN};function MN(e){let{inputs:t,backend:n,attrs:r}=e,{logits:i}=t,{dim:a}=r,o=D([a],i.shape),s=nN({inputs:{x:i},backend:n,attrs:{reductionIndices:o,keepDims:!1}}),c=El(s.shape,o),l=$({inputs:{x:s},backend:n,attrs:{shape:c}}),u=AN({inputs:{a:i,b:l},backend:n}),d=$j({inputs:{x:u},backend:n}),f=$O({inputs:{x:d},backend:n,attrs:{axis:o,keepDims:!1}}),p=$({inputs:{x:f},backend:n,attrs:{shape:c}}),m=DN({inputs:{a:d,b:p},backend:n});return n.disposeIntermediateTensorInfo(s),n.disposeIntermediateTensorInfo(l),n.disposeIntermediateTensorInfo(u),n.disposeIntermediateTensorInfo(d),n.disposeIntermediateTensorInfo(f),n.disposeIntermediateTensorInfo(p),m}var NN={kernelName:Qn,backendName:`webgl`,kernelFunc:MN};function PN(e){let{inputs:t,backend:n,attrs:r}=e,{logits:i}=t,{numSamples:a,seed:o,normalized:s}=r,c=s?i:MN({inputs:{logits:i},backend:n,attrs:{dim:i.shape.length-1}}),l=c.shape[0],u=c.shape[1],d=new EN(l,u,a),f=[[o]],p=n.runWebGLProgram(d,[c],`int32`,f);return s||n.disposeIntermediateTensorInfo(c),p}var FN={kernelName:dn,backendName:`webgl`,kernelFunc:PN},IN=KD+`
  return -x;
`,LN=`
  vec4 result = -x;
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`;function RN(e){let{inputs:t,backend:n}=e,{x:r}=t;if(n.shouldExecuteOnCPU([r])){let[e,t]=oD(n.texData.get(r.dataId).values,r.shape,r.dtype);return n.makeTensorInfo(t,r.dtype,e)}let i;return i=O().getBool(`WEBGL_PACK_UNARY_OPERATIONS`)?new aO(r.shape,LN):new GD(r.shape,IN),n.runWebGLProgram(i,[r],r.dtype)}var zN={kernelName:`Neg`,backendName:`webgl`,kernelFunc:RN},BN=Th;function VN(e){Er(`tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead`);let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c}=r,{selectedIndices:l}=BN(n.readSync(i.dataId),n.readSync(a.dataId),o,s,c);return n.makeTensorInfo([l.length],`int32`,new Int32Array(l))}var HN={kernelName:mn,backendName:`webgl`,kernelFunc:VN},UN=Eh;function WN(e){Er(`tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead`);let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c,padToMaxOutputSize:l}=r,{selectedIndices:u,validOutputs:d}=UN(n.readSync(i.dataId),n.readSync(a.dataId),o,s,c,l);return[n.makeTensorInfo([u.length],`int32`,new Int32Array(u)),n.makeTensorInfo([],`int32`,new Int32Array([d]))]}var GN={kernelName:hn,backendName:`webgl`,kernelFunc:WN},KN=Dh;function qN(e){Er(`tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead`);let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c,softNmsSigma:l}=r,{selectedIndices:u,selectedScores:d}=KN(n.readSync(i.dataId),n.readSync(a.dataId),o,s,c,l);return[n.makeTensorInfo([u.length],`int32`,new Int32Array(u)),n.makeTensorInfo([d.length],`float32`,new Float32Array(d))]}var JN={kernelName:gn,backendName:`webgl`,kernelFunc:qN},YN=class{constructor(e,t,n,r){this.variableNames=[`indices`],this.outputShape=[e,t],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int index = round(getIndices(coords.x));
        setOutput(mix(float(${r}), float(${n}),
                      float(index == coords.y)));
      }
    `}},XN={kernelName:vn,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{indices:i}=t,{dtype:a,depth:o,onValue:s,offValue:c}=r,l=S(i.shape),u=new YN(l,o,s,c),d=$({inputs:{x:i},backend:n,attrs:{shape:[l]}}),f=n.runWebGLProgram(u,[d],a);n.disposeIntermediateTensorInfo(d);let p=[...i.shape,o],m=$({inputs:{x:f},backend:n,attrs:{shape:p}});return n.disposeIntermediateTensorInfo(f),m}};function ZN(e){let{inputs:t,backend:n}=e,{x:r}=t;if(r.dtype===`complex64`){let e=gA({inputs:{input:r},backend:n}),t=ZN({inputs:{x:e},backend:n}),i=FA({inputs:{input:r},backend:n}),a=ZN({inputs:{x:i},backend:n}),o=CO({inputs:{real:t,imag:a},backend:n});return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),n.disposeIntermediateTensorInfo(i),n.disposeIntermediateTensorInfo(a),o}else return uM({attrs:{shape:r.shape,dtype:r.dtype,value:r.dtype===`string`?``:0},backend:n})}var QN={kernelName:yr,backendName:`webgl`,kernelFunc:ZN};function $N(e){let{inputs:t,backend:n}=e,{x:r}=t;if(r.dtype===`string`)throw Error(`onesLike is not supported under string dtype`);if(r.dtype===`complex64`){let e=gA({inputs:{input:r},backend:n}),t=$N({inputs:{x:e},backend:n}),i=FA({inputs:{input:r},backend:n}),a=ZN({inputs:{x:i},backend:n}),o=CO({inputs:{real:t,imag:a},backend:n});return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),n.disposeIntermediateTensorInfo(i),n.disposeIntermediateTensorInfo(a),o}else return uM({attrs:{shape:r.shape,dtype:r.dtype,value:1},backend:n})}var eP={kernelName:_n,backendName:`webgl`,kernelFunc:$N};function tP(e){let{inputs:t,backend:n,attrs:r}=e,{axis:i}=r;if(t.length===1)return tM({inputs:{input:t[0]},backend:n,attrs:{dim:i}});let a=t[0].shape,o=t[0].dtype;t.forEach(e=>{b(a,e.shape,`All tensors passed to stack must have matching shapes`),y(o===e.dtype,()=>`All tensors passed to stack must have matching dtypes`)});let s=[],c=zA({inputs:t.map(e=>{let t=tM({inputs:{input:e},backend:n,attrs:{dim:i}});return s.push(t),t}),backend:n,attrs:{axis:i}});return s.forEach(e=>n.disposeIntermediateTensorInfo(e)),c}var nP={kernelName:yn,backendName:`webgl`,kernelFunc:tP},rP=class{constructor(e,t,n){this.variableNames=[`x`],this.customUniforms=[{name:`value`,type:`float`}],this.outputShape=t.map((t,n)=>t[0]+e[n]+t[1]);let r=e.length,i=Z(r),a=t.map(e=>e[0]).join(`,`),o=t.map((t,n)=>t[0]+e[n]).join(`,`),s=[`coords[0]`,`coords[1]`,`coords[2]`,`coords[3]`].slice(0,r);if(r===1){this.userCode=`
        int start = ${a};
        int end = ${o};

        void main() {
          int outC = getOutputCoords();
          if (outC < start || outC >= end) {
            setOutput(value);
          } else {
            setOutput(getX(outC - start));
          }
        }
      `;return}this.userCode=`
      ${i} start = ${i}(${a});
      ${i} end = ${i}(${o});

      void main() {
        ${i} outC = getOutputCoords();
        if (any(lessThan(outC, start)) || any(greaterThanEqual(outC, end))) {
          setOutput(value);
        } else {
          ${i} coords = outC - start;
          setOutput(getX(${s}));
        }
      }
    `}},iP=class{constructor(e,t,n){this.variableNames=[`x`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`value`,type:`float`}],this.outputShape=t.map((t,n)=>t[0]+e[n]+t[1]);let r=e.length,i=Z(r),a=t.map(e=>e[0]).join(`,`),o=t.map((t,n)=>t[0]+e[n]).join(`,`),s=ND(`rc`,r),c=ND(`source`,r),l=`${s[r-1]} < ${this.outputShape[r-1]}`,u=r===1?`source`:`vec2(${c.slice(-2).join()})`,d=[`${i} rc = outputLoc;`,`${s[r-1]} += 1;
       if(${l}) {
      `,r===1?``:`}
       rc = outputLoc;
       ${s[r-2]} += 1;
       if(${s[r-2]} < ${this.outputShape[r-2]}) {`,r===1?``:`  ${s[r-1]} += 1;
         if(${l}) {`],f=r===1?`rc < start || rc >= end`:`any(lessThan(rc, start)) || any(greaterThanEqual(rc, end))`,p=``;for(let e=0,t=r===1?2:4;e<t;e++)p+=`
        ${d[e]}
        if (${f}) {
          result[${e}] = float(value);
        } else {
          ${i} source = rc - start;
          result[${e}] = getChannel(getX(${c.join()}), ${u});
        }
      `;p+=r===1?`} `:`}}`,this.userCode=`
      const ${i} start = ${i}(${a});
      const ${i} end = ${i}(${o});

      void main() {
        ${i} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${p}
        setOutput(result);
      }
    `}},aP=e=>{let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{paddings:a,constantValue:o}=r;if(S(i.shape)===0)return uM({backend:n,attrs:{shape:a.map((e,t)=>e[0]+i.shape[t]+e[1]),value:o,dtype:i.dtype}});let s=O().getBool(`WEBGL_PACK_ARRAY_OPERATIONS`)?new iP(i.shape,a,o):new rP(i.shape,a,o),c=[[o]];return n.runWebGLProgram(s,[i],i.dtype,c)},oP={kernelName:bn,backendName:`webgl`,kernelFunc:aP},sP={kernelName:`Pow`,backendName:`webgl`,kernelFunc:PO({opSnippet:`
  if(a < 0.0 && floor(b) < b){
    return NAN;
  }
  if (b == 0.0) {
    return 1.0;
  }
  return (round(mod(b, 2.0)) != 1) ?
      pow(abs(a), b) : sign(a) * pow(abs(a), b);
`,packedOpSnippet:`
  // isModRound1 has 1 for components with round(mod(b, 2.0)) == 1, 0 otherwise.
  vec4 isModRound1 = vec4(equal(round(mod(b, 2.0)), ivec4(1)));
  vec4 multiplier = sign(a) * isModRound1 + (vec4(1.0) - isModRound1);
  vec4 result = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  bvec4 isExpZero = equal(b, vec4(0.0));
  result.r = isExpZero.r ? 1.0 : result.r;
  result.g = isExpZero.g ? 1.0 : result.g;
  result.b = isExpZero.b ? 1.0 : result.b;
  result.a = isExpZero.a ? 1.0 : result.a;

  bvec4 isNaN1 = lessThan(a, vec4(0.0));
  bvec4 isNaN2 = lessThan(floor(b), b);
  bvec4 isNaN = bvec4(isNaN1.x && isNaN2.x, isNaN1.y && isNaN2.y, isNaN1.z && isNaN2.z, isNaN1.w && isNaN2.w);
  `+yO+`
  return result;
`})};function cP(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r,s=i.shape.length,c=[],l=D(a,i.shape),u=l,d=Ol(u,s),f=i;d!=null&&(f=tk({inputs:{x:i},backend:n,attrs:{perm:d}}),u=Al(u.length,s),c.push(f)),Dl(`prod`,u,s);let p;if(n.shouldExecuteOnCPU([f])){let e=n.texData.get(f.dataId).values,{outVals:t,outShape:r,outDtype:i}=cD(f.shape,f.dtype,e,u);p=n.makeTensorInfo(r,i,t)}else{let[e,t]=Tl(f.shape,u),r=S(t),a=$({inputs:{x:f},backend:n,attrs:{shape:[-1,r]}}),o=qO(a,Ii(i.dtype),`prod`,n);p=$({inputs:{x:o},backend:n,attrs:{shape:e}}),c.push(a),c.push(o)}if(o){c.push(p);let e=El(p.shape,l);p=$({inputs:{x:p},backend:n,attrs:{shape:e}})}return c.forEach(e=>n.disposeIntermediateTensorInfo(e)),p}var lP={kernelName:Sn,backendName:`webgl`,kernelFunc:cP};function uP(e){let{inputs:t,backend:n,attrs:r}=e,{paramsNestedSplits:i,paramsDenseValues:a,indices:o}=t,{outputRaggedRank:s}=r,c=i.map(e=>n.readSync(e.dataId)),l=i.map(e=>e.shape),u=n.readSync(a.dataId),d=n.readSync(o.dataId),[f,p,m]=lD(c,l,u,a.shape,a.dtype,d,o.shape,s),h=f.map(e=>n.makeTensorInfo([e.length],`int32`,e)),g=n.makeTensorInfo(m,a.dtype,p);return h.concat([g])}var dP={kernelName:Cn,backendName:`webgl`,kernelFunc:uP};function fP(e){let{inputs:t,backend:n}=e,{starts:r,limits:i,deltas:a}=t,o=n.readSync(r.dataId),s=n.readSync(i.dataId),c=n.readSync(a.dataId),[l,u]=uD(o,r.shape,r.dtype,s,i.shape,c,a.shape);return[n.makeTensorInfo([l.length],`int32`,l),n.makeTensorInfo([u.length],r.dtype,u)]}var pP={kernelName:wn,backendName:`webgl`,kernelFunc:fP};function mP(e){let{inputs:t,backend:n,attrs:r}=e,{shape:i,values:a,defaultValue:o,rowPartitionTensors:s}=t,{rowPartitionTypes:c}=r,l=n.readSync(i.dataId),u=n.readSync(a.dataId),d=n.readSync(o.dataId),f=s.map(e=>n.readSync(e.dataId)),p=s.map(e=>e.shape),[m,h]=dD(l,i.shape,u,a.shape,a.dtype,d,o.shape,f,p,c);return n.makeTensorInfo(m,a.dtype,h)}var hP={kernelName:Tn,backendName:`webgl`,kernelFunc:mP},gP=e=>{let{backend:t,attrs:n}=e,{start:r,stop:i,step:a,dtype:o}=n,s=fD(r,i,a,o);return t.makeTensorInfo([s.length],o,s)},_P={kernelName:En,backendName:`webgl`,kernelFunc:gP},vP={kernelName:On,backendName:`webgl`,kernelFunc:Q({opSnippet:`return 1.0 / x;`})},yP={kernelName:kn,backendName:`webgl`,kernelFunc:Q({opSnippet:KD+`
  return (x < 0.0) ? 0.0 : x;
`,packedOpSnippet:`
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`})},bP={kernelName:Fn,backendName:`webgl`,kernelFunc:Q({opSnippet:KD+`
  return (x < 0.0) ? 0.0 : min(6.0, x);
`,packedOpSnippet:`
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`})},xP=class{constructor(e,t,n,r,i){this.variableNames=[`A`],this.outputShape=[];let[a,o,s,c]=e;this.outputShape=[a,t,n,c];let l=[r&&t>1?o-1:o,r&&n>1?s-1:s],u=[r&&t>1?t-1:t,r&&n>1?n-1:n],d;d=i?`(vec2(yRC) + vec2(0.5)) * effectiveInputOverOutputRatioRC - vec2(0.5)`:`vec2(yRC) * effectiveInputOverOutputRatioRC`,this.userCode=`
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${l[0]/u[0]},
          ${l[1]/u[1]});
      const vec2 inputShapeRC = vec2(${o}.0, ${s}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = ${d};

        // Compute the four integer indices.
        ivec2 sourceFloorRC = ivec2(max(sourceFracIndexRC, vec2(0.0)));
        ivec2 sourceCeilRC = ivec2(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        float topLeft = getA(b, sourceFloorRC.x, sourceFloorRC.y, d);
        float bottomLeft = getA(b, sourceCeilRC.x, sourceFloorRC.y, d);
        float topRight = getA(b, sourceFloorRC.x, sourceCeilRC.y, d);
        float bottomRight = getA(b, sourceCeilRC.x, sourceCeilRC.y, d);

        vec2 fracRC = sourceFracIndexRC - vec2(sourceFloorRC);

        float top = topLeft + (topRight - topLeft) * fracRC.y;
        float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
        float newValue = top + (bottom - top) * fracRC.x;

        setOutput(newValue);
      }
    `}},SP=class{constructor(e,t,n,r,i){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];let[a,o,s,c]=e;this.outputShape=[a,t,n,c];let l=[r&&t>1?o-1:o,r&&n>1?s-1:s],u=[r&&t>1?t-1:t,r&&n>1?n-1:n],d;d=i?`(vec3(yRC) + vec3(0.5)) * effectiveInputOverOutputRatioRC - vec3(0.5)`:`vec3(yRC) * effectiveInputOverOutputRatioRC`,this.userCode=`
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${l[0]/u[0]},
          ${l[1]/u[1]},
          ${l[1]/u[1]});
      const vec3 inputShapeRC = vec3(${o}.0, ${s}.0,
                                     ${s}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${d};

        // Compute the four integer indices.
        ivec3 sourceFloorRC = ivec3(max(sourceFracIndexRC, vec3(0.0)));
        ivec3 sourceCeilRC = ivec3(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${c-1};
        bool hasNextRow = coords.z < ${n-1};

        // In parallel, construct four corners for all four components in
        // packed 2x2 cell.
        vec4 topLeft = vec4(
          getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 bottomLeft = vec4(
          getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 topRight = vec4(
          getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec4 bottomRight = vec4(
          getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec3 fracRC = sourceFracIndexRC - vec3(sourceFloorRC);

        vec4 top = mix(topLeft, topRight, fracRC.yyzz);
        vec4 bottom = mix(bottomLeft, bottomRight, fracRC.yyzz);
        vec4 newValue = mix(top, bottom, fracRC.x);

        setOutput(newValue);
      }
    `}};function CP(e){let{inputs:t,backend:n,attrs:r}=e,{images:i}=t,{alignCorners:a,halfPixelCenters:o,size:s}=r,[c,l]=s,u=O().getBool(`WEBGL_PACK_IMAGE_OPERATIONS`)?new SP(i.shape,c,l,a,o):new xP(i.shape,c,l,a,o);return n.runWebGLProgram(u,[i],`float32`)}var wP={kernelName:Nn,backendName:`webgl`,kernelFunc:CP},TP=class{constructor(e,t,n){this.variableNames=[`dy`],this.outputShape=[],this.outputShape=t;let[,r,i]=t,[,a,o]=e,s=[n&&a>1?r-1:r,n&&o>1?i-1:i],c=[n&&a>1?a-1:a,n&&o>1?o-1:o],l=s[0]/c[0],u=s[1]/c[1],d=1/l,f=1/u,p=Math.ceil(d)*2+2,m=Math.ceil(f)*2+2;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${l});
        const float widthScale = float(${u});

        const float invHeightScale = float(${d});
        const float invWidthScale = float(${f});

        const int winHeight = int(${p});
        const int winWidth = int(${m});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(startRLerp - float(winHeight / 2));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(startCLerp - float(winWidth / 2));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${a}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${o}) {
              continue;
            }

            float dxR = float(dyR) * heightScale;
            int topDxRIndex = int(floor(dxR));
            int bottomDxRIndex = int(min(ceil(dxR), ${r-1}.0));
            float dxRLerp = dxR - float(topDxRIndex);
            float inverseDxRLerp = 1.0 - dxRLerp;

            float dxC = float(dyC) * widthScale;
            int leftDxCIndex = int(floor(dxC));
            int rightDxCIndex = int(min(ceil(dxC), ${i-1}.0));
            float dxCLerp = dxC - float(leftDxCIndex);
            float inverseDxCLerp = 1.0 - dxCLerp;

            if (r == topDxRIndex && c == leftDxCIndex) {
              // topLeft
              accumulator +=
                getDy(b, dyR, dyC, d) * inverseDxRLerp * inverseDxCLerp;
            }

            if (r == topDxRIndex && c == rightDxCIndex) {
              // topRight
              accumulator += getDy(b, dyR, dyC, d) * inverseDxRLerp * dxCLerp;
            }

            if (r == bottomDxRIndex && c == leftDxCIndex) {
              // bottomLeft
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * inverseDxCLerp;
            }

            if (r == bottomDxRIndex && c == rightDxCIndex) {
              // bottomRight
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * dxCLerp;
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `}};function EP(e){let{inputs:t,backend:n,attrs:r}=e,{images:i,dy:a}=t,{alignCorners:o}=r,s=new TP(a.shape,i.shape,o);return n.runWebGLProgram(s,[a],a.dtype)}var DP={kernelName:Pn,backendName:`webgl`,kernelFunc:EP},OP=class{constructor(e,t,n,r,i){this.variableNames=[`A`],this.outputShape=[];let[a,o,s,c]=e;this.outputShape=[a,t,n,c];let l=[r&&t>1?o-1:o,r&&n>1?s-1:s],u=[r&&t>1?t-1:t,r&&n>1?n-1:n],d=r?`0.5`:`0.0`,f;f=i?`max((vec2(yRC) + vec2(0.5)) * effectiveInputOverOutputRatioRC, vec2(0.0))`:`vec2(yRC) * effectiveInputOverOutputRatioRC`,this.userCode=`
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${l[0]/u[0]},
          ${l[1]/u[1]});
      const vec2 inputShapeRC = vec2(${o}.0, ${s}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = ${f};

        // Compute the coordinators of nearest neighbor point.
        ivec2 sourceNearestRC = ivec2(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${d})));
        float newValue = getA(b, sourceNearestRC.x, sourceNearestRC.y, d);

        setOutput(newValue);
      }
    `}},kP=class{constructor(e,t,n,r,i){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];let[a,o,s,c]=e;this.outputShape=[a,t,n,c];let l=[r&&t>1?o-1:o,r&&n>1?s-1:s],u=[r&&t>1?t-1:t,r&&n>1?n-1:n],d=r?`0.5`:`0.0`,f;f=i?`max((vec3(yRC) + vec3(0.5)) * effectiveInputOverOutputRatioRC, vec3(0.0))`:`vec3(yRC) * effectiveInputOverOutputRatioRC`,this.userCode=`
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${l[0]/u[0]},
          ${l[1]/u[1]},
          ${l[1]/u[1]});
      const vec3 inputShapeRC = vec3(${o}.0, ${s}.0,
                                     ${s}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${f};

        // Compute the coordinators of nearest neighbor point.
        ivec3 sourceNearestRC = ivec3(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${d})));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${c-1};
        bool hasNextRow = coords.z < ${n-1};

        vec4 newValue = vec4(
          getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d),
          hasNextCol ? getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d + 1) : 0.0);

        setOutput(newValue);
      }
    `}};function AP(e){let{inputs:t,backend:n,attrs:r}=e,{images:i}=t,{alignCorners:a,halfPixelCenters:o,size:s}=r,[c,l]=s,u=O().getBool(`WEBGL_PACK_IMAGE_OPERATIONS`)?new kP(i.shape,c,l,a,o):new OP(i.shape,c,l,a,o);return n.runWebGLProgram(u,[i],i.dtype)}var jP={kernelName:jn,backendName:`webgl`,kernelFunc:AP},MP=class{constructor(e,t,n){this.variableNames=[`dy`],this.outputShape=[],this.outputShape=t;let[,r,i]=t,[,a,o]=e,s=[n&&a>1?r-1:r,n&&o>1?i-1:i],c=[n&&a>1?a-1:a,n&&o>1?o-1:o],l=s[0]/c[0],u=s[1]/c[1],d=1/l,f=1/u,p=Math.ceil(d)*2+2,m=Math.ceil(f)*2+2;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${l});
        const float widthScale = float(${u});

        const float invHeightScale = float(${d});
        const float invWidthScale = float(${f});

        const int winHeight = int(${p});
        const int winWidth = int(${m});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(floor(startRLerp - float(winHeight / 2)));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(floor(startCLerp - float(winWidth / 2)));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${a}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${o}) {
              continue;
            }

            float sourceFracRow =
              float(${s[0]}) *
                (float(dyR) / float(${c[0]}));

            float sourceFracCol =
                float(${s[1]}) *
                  (float(dyC) / float(${c[1]}));

            int sourceNearestRow = int(min(
                float(int(${r}) - 1),
                ${n} ? float(round(sourceFracRow)) :
                                  float(floor(sourceFracRow))));

            int sourceNearestCol = int(min(
                float(int(${i}) - 1),
                ${n} ? float(round(sourceFracCol)) :
                                  float(floor(sourceFracCol))));

            if (r == sourceNearestRow && c == sourceNearestCol) {
              accumulator += getDy(b, dyR, dyC, d);
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `}};function NP(e){let{inputs:t,backend:n,attrs:r}=e,{images:i,dy:a}=t,{alignCorners:o}=r,s=new MP(a.shape,i.shape,o);return n.runWebGLProgram(s,[a],a.dtype)}var PP={kernelName:Mn,backendName:`webgl`,kernelFunc:NP},FP=class{constructor(e,t){this.variableNames=[`x`];let n=e.length;if(n>4)throw Error(`WebGL backend: Reverse of rank-${n} tensor is not yet supported`);if(this.outputShape=e,n===1){this.userCode=`
        void main() {
          int coord = getOutputCoords();
          setOutput(getX(${e[0]} - coord - 1));
        }
      `;return}let r=n=>t.indexOf(n)!==-1&&e[n]!==1?`${e[n]} - coords[${n}] - 1`:`coords[${n}]`,i=e.map((e,t)=>r(t)).join(`,`),a=Z(n);this.userCode=`
      void main() {
        ${a} coords = getOutputCoords();
        setOutput(getX(${i}));
      }
    `}},IP=class{constructor(e,t){this.variableNames=[`x`],this.packedInputs=!0,this.packedOutput=!0;let n=e.length;if(n>4)throw Error(`WebGL backend: Reverse of rank-${n} tensor is not yet supported`);this.outputShape=e;let r=ND(`rc`,n),i=`${r[n-1]} + 1 < ${this.outputShape[n-1]}`,a=`${r[n-2]} + 1 < ${this.outputShape[n-2]}`,o=Z(n);n===1?this.userCode=`
        void main(){
          int rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = getChannel(getX(${e[0]} - rc - 1),
            ${e[0]} - rc - 1);
          if(${i}){
              result.g = getChannel(getX(${e[0]} - (rc  + 1) - 1),
                ${e[0]} - (rc  + 1) - 1);
          }
          setOutput(result);
        }
      `:this.userCode=`
        void main() {
          ${o} rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = ${s(r.slice())};
          if(${i}){
            result.g = ${c(r.slice())};
          }
          if(${a}) {
            result.b = ${l(r.slice())};
            if(${i}) {
              result.a = ${u(r.slice())};
            }
          }
          setOutput(result);
        }
    `;function s(e){return d(e)}function c(e){return e[n-1]=`(`+e[n-1]+` + 1)`,d(e)}function l(e){return e[n-2]=`(`+e[n-2]+` + 1)`,d(e)}function u(e){return e[n-1]=`(`+e[n-1]+` + 1)`,e[n-2]=`(`+e[n-2]+` + 1)`,d(e)}function d(t){let n=e.map((e,n)=>f(n,t));return`getChannel(getX(${n.join(`,`)}), vec2(${n.slice(-2).join(`,`)}))`}function f(n,r){return t.indexOf(n)!==-1&&e[n]!==1?`${e[n]} - ${r[n]} - 1`:`${r[n]}`}}};function LP(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{dims:a}=r,o=i.shape.length,s=D(a,i.shape);if(o===0)return xO({inputs:{x:i},backend:n});let c=O().getBool(`WEBGL_PACK_ARRAY_OPERATIONS`)?new IP(i.shape,s):new FP(i.shape,s);return n.runWebGLProgram(c,[i],i.dtype)}var RP={kernelName:In,backendName:`webgl`,kernelFunc:LP},zP=class{constructor(e,t){this.variableNames=[`Image`],this.outputShape=[],this.customUniforms=[{name:`params`,type:`vec4`}];let n=e[1],r=e[2];this.outputShape=e;let i=``;i=typeof t==`number`?`float outputValue = ${t.toFixed(2)};`:`
        vec3 fill = vec3(${t.join(`,`)});
        float outputValue = fill[coords[3]];`,this.userCode=`
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];
          int y = coords[1];
          float coordXFloat = (float(x) - params[0]) * params[3] -
            (float(y) - params[1]) * params[2];
          float coordYFloat = (float(x) - params[0]) * params[2] +
            (float(y) - params[1]) * params[3];
          int coordX = int(round(coordXFloat + params[0]));
          int coordY = int(round(coordYFloat + params[1]));
          ${i}
          if(coordX >= 0 && coordX < ${r} && coordY >= 0 && coordY < ${n}) {
            outputValue = getImage(coords[0], coordY, coordX, coords[3]);
          }
          setOutput(outputValue);
        }
    `}},BP={kernelName:Sr,backendName:`webgl`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{image:r}=e,{radians:i,fillValue:a,center:o}=t,s=n,c=new zP(r.shape,a),[l,u]=bv(o,r.shape[1],r.shape[2]),d=[[l,u,Math.sin(i),Math.cos(i)]];return s.runWebGLProgram(c,[r],r.dtype,d)}},VP={kernelName:Ln,backendName:`webgl`,kernelFunc:Q({opSnippet:`
  // OpenGL ES does not support round function.
  // The algorithm is based on banker's rounding.
  float base = floor(x);
  if ((x - base) < 0.5) {
    return floor(x);
  } else if ((x - base) > 0.5) {
    return ceil(x);
  } else {
    if (mod(base, 2.0) == 0.0) {
      return base;
    } else {
      return base + 1.0;
    }
  }
`})},HP={kernelName:Rn,backendName:`webgl`,kernelFunc:Q({opSnippet:`return inversesqrt(x);`,cpuKernelImpl:pD})},UP=class{constructor(e,t,n,r,i,a,o=!0,s=!1){this.variableNames=[`updates`,`indices`,`defaultValue`],this.outputShape=a;let c=Z(i.length),l=Z(a.length),u=``;n===1?u=`i`:n===2&&(u=`i, j`);let d=`getIndices(${u})`,f=``;r===1?f=`i`:r===2&&(f=`i, coords[1]`);let p=`getUpdates(${f})`,m=``;s&&(m=`coords[0], coords[1]`);let h=`getDefaultValue(${m})`,g=t>1?`strides[j]`:`strides`;this.userCode=`
        ${c} strides = ${c}(${i});

        void main() {
          ${l} coords = getOutputCoords();
          float sum = 0.0;
          bool found = false;
          for (int i = 0; i < ${e}; i++) {
            int flattenedIndex = 0;
            for (int j = 0; j < ${t}; j++) {
              int index = round(${d});
              flattenedIndex += index * ${g};
            }
            if (flattenedIndex == coords[0]) {
              sum += ${p};
              found = true;
            }
          }
          setOutput(mix(${h}, sum, float(found)));
        }
      `}},WP=class{constructor(e,t,n,r,i,a,o=!0,s=!1){this.variableNames=[`updates`,`indices`,`defaultValue`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=a;let c=Z(i.length),l=Z(a.length),u=``;n===1?u=`i`:n===2&&(u=`i, j`);let d=`getIndices(${u})`,f=``;r===1?f=`i`:r===2&&(f=`i, coords[1]`);let p=`getUpdates(${f})`,m=``;s&&(m=`coords[0], coords[1]`);let h=`getDefaultValue(${m})`,g=t>1?`strides[j]`:`strides`,_=t>1?`strides[j + 1]`:`strides`;this.userCode=`
        ${c} strides = ${c}(${i});

        void main() {
          ${l} coords = getOutputCoords();
          vec4 sum = vec4(0.);
          vec4 found = vec4(0.);
          for (int i = 0; i < ${e}; i+=2) {
            ivec2 flattenedIndex = ivec2(0);
            for (int j = 0; j < ${t}; j+=2) {
              ivec4 index = round(${d});
              flattenedIndex += index.xz * ${g};
              if (j + 1 < ${t}) {
                flattenedIndex += index.yw * ${_};
              }
            }
            if (flattenedIndex[0] == coords[0] || flattenedIndex[1] == coords[0] ||
                flattenedIndex[0] == coords[0] + 1 || flattenedIndex[1] == coords[0] + 1) {
              vec4 updVals = ${p};
              if (flattenedIndex[0] == coords[0]) {
                sum.xy += updVals.xy;
                found.xy = vec2(1.);
              } else if (flattenedIndex[0] == coords[0] + 1) {
                sum.zw += updVals.xy;
                found.zw = vec2(1.);
              }
              if (flattenedIndex[1] == coords[0]) {
                sum.xy += updVals.zw;
                found.xy = vec2(1.);
              } else if (flattenedIndex[1] == coords[0] + 1) {
                sum.zw += updVals.zw;
                found.zw = vec2(1.);
              }
            }
          }
          setOutput(mix(${h}, sum, found));
        }
      `}};function GP(e){let{inputs:t,backend:n,attrs:r}=e,{indices:i,updates:a}=t,{shape:o}=r,{sliceRank:s,numUpdates:c,sliceSize:l,strides:u,outputSize:d}=$p(a,i,o),f=[d/l,l];if(d===0)return n.makeTensorInfo(o,i.dtype);let p=$({inputs:{x:i},backend:n,attrs:{shape:[c,s]}}),m=$({inputs:{x:a},backend:n,attrs:{shape:[c,l]}}),h=n.makeTensorInfo([],`float32`,new Float32Array([0])),g;g=O().getBool(`WEBGL_PACK`)?new WP(c,s,p.shape.length,m.shape.length,u,f):new UP(c,s,p.shape.length,m.shape.length,u,f);let _=n.runWebGLProgram(g,[m,p,h],m.dtype),v=$({inputs:{x:_},backend:n,attrs:{shape:o}});return n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(_),n.disposeIntermediateTensorInfo(h),v}var KP={kernelName:zn,backendName:`webgl`,kernelFunc:GP},qP=class{constructor(e,t,n,r){this.variableNames=[`sortedSequence`,`values`],this.customUniforms=[{name:`numInputs`,type:`int`}],this.outputShape=[e,n];let i=`for (int i = 0; i < ${Math.ceil(Math.log2(t+1))}; ++i) { if (left >= right) break;`,a=O().getNumber(`WEBGL_VERSION`)===2?`while (left < right) {`:i,o=r===`left`?`<`:`<=`;this.userCode=`
       int findBound(int batch, float value) {
         int left = 0;
         int right = numInputs;
         int mid;
         ${a}
           mid = (left + right) / 2;
           if (getSortedSequence(batch, mid) ${o} value) {
             left = mid + 1;
           } else {
             right = mid;
           }
         }
         return right;
       }

       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int valueIndex = coords[1];

         float value = getValues(batch, valueIndex);

         setOutput(float(findBound(batch, value)));
       }
     `}};function JP(e){let{inputs:t,backend:n,attrs:r}=e,{sortedSequence:i,values:a}=t,{side:o}=r,s=new qP(i.shape[0],i.shape[1],a.shape[1],o),c=[[i.shape[1]]];return n.runWebGLProgram(s,[i,a],`int32`,c)}var YP={kernelName:Vn,backendName:`webgl`,kernelFunc:JP},XP=class{constructor(e,t,n){this.variableNames=[`c`,`a`,`b`],this.outputShape=t;let r,i;if(n>4)throw Error(`Where for rank ${n} is not yet supported`);if(n===1)i=`resRC`,r=`resRC`;else{let n=[`resRC.x`,`resRC.y`,`resRC.z`,`resRC.w`],a=[],o=[];for(let r=0;r<t.length;r++)o.push(`${n[r]}`),r<e&&a.push(`${n[r]}`);r=a.join(),i=o.join()}let a=Z(n);this.userCode=`
      void main() {
        ${a} resRC = getOutputCoords();
        float cVal = getC(${r});
        if (cVal >= 1.0) {
          setOutput(getA(${i}));
        } else {
          setOutput(getB(${i}));
        }
      }
    `}};function ZP(e){let{inputs:t,backend:n}=e,{condition:r,t:i,e:a}=t,o=new XP(r.shape.length,i.shape,i.shape.length);return n.runWebGLProgram(o,[r,i,a],Fi(i.dtype,a.dtype))}var QP={kernelName:Hn,backendName:`webgl`,kernelFunc:ZP},$P={kernelName:Un,backendName:`webgl`,kernelFunc:Q({opSnippet:`
  // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
  // see: https://arxiv.org/abs/1706.02515
  float scaleAlpha = ${Ev};
  float scale = ${Dv};
  return (x >= 0.0) ? scale * x : scaleAlpha * (exp(x) - 1.0);
`})},eF={kernelName:qn,backendName:`webgl`,kernelFunc:Q({opSnippet:NO+`
  return 1.0 / (1.0 + exp(-1.0 * x));
`,packedOpSnippet:`
  vec4 result = 1.0 / (1.0 + exp(-1.0 * x));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,cpuKernelImpl:hD})},tF={kernelName:Kn,backendName:`webgl`,kernelFunc:Q({opSnippet:`
  if (isnan(x)) { return 0.0; }
  return sign(x);
`})},nF={kernelName:`Sin`,backendName:`webgl`,kernelFunc:Q({opSnippet:NO+`
  return sin(x);
`,packedOpSnippet:`
  vec4 result = sin(x);
  bvec4 isNaN = isnan(x);
  ${yO}
  return result;
`})},rF={kernelName:Gn,backendName:`webgl`,kernelFunc:Q({opSnippet:`
  float e2x = exp(x);
  return (e2x - 1.0 / e2x) / 2.0;
`})},iF={kernelName:Jn,backendName:`webgl`,kernelFunc:Q({opSnippet:`
  float epsilon = 1.1920928955078125e-7;
  float threshold = log(epsilon) + 2.0;

  bool too_large = x > -threshold;
  bool too_small = x < threshold;

  float result;
  float exp_x = exp(x);

  if (too_large){
    result = x;
  }
  else if (too_small){
    result = exp_x;
  }
  else{
    result = log(exp_x + 1.0);
  }
  return result;
`})},aF={kernelName:Xn,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockShape:a,paddings:o}=r;y(i.shape.length<=4,()=>`spaceToBatchND for rank > 4 with a WebGL backend not implemented yet`);let s=a.reduce((e,t)=>e*t),c=[[0,0]];c.push(...o);for(let e=1+a.length;e<i.shape.length;++e)c.push([0,0]);let l=[],u=aP({inputs:{x:i},backend:n,attrs:{paddings:c,constantValue:0}}),d=xv(u.shape,a,s,!1),f=Sv(d.length,a.length,!1),p=Cv(u.shape,a,s,!1),m=$({inputs:{x:u},backend:n,attrs:{shape:d}}),h=tk({inputs:{x:m},backend:n,attrs:{perm:f}}),g=$({inputs:{x:h},backend:n,attrs:{shape:p}});return l.push(u),l.push(m),l.push(h),l.forEach(e=>n.disposeIntermediateTensorInfo(e)),g}};function oF(e){let{inputs:t,backend:n}=e,{indices:r,values:i,denseShape:a,defaultValue:o}=t;if(a.shape.length!==1)throw Error(`Dense shape must be a vector, saw:
         ${a.shape}`);if(r.shape.length!==2)throw Error(`Indices must be a matrix, saw:
         ${r.shape}`);if(i.shape.length!==1)throw Error(`Values must be a vector, saw:
         ${i.shape}`);if(o.shape.length!==0)throw Error(`Default value must be a scalar, saw:
        ${o.shape}`);let s=n.readSync(r.dataId),c=n.readSync(i.dataId),l=n.readSync(a.dataId),u=n.readSync(o.dataId)[0],[d,f,p,m,h]=vD(s,r.shape,r.dtype,c,i.dtype,l,u);return[n.makeTensorInfo(f,r.dtype,d),n.makeTensorInfo([f[0]],i.dtype,p),n.makeTensorInfo([m.length],`bool`,new Uint8Array(m.map(e=>Number(e)))),n.makeTensorInfo([h.length],r.dtype,new Int32Array(h))]}var sF={kernelName:$n,backendName:`webgl`,kernelFunc:oF};function cF(e){let{inputs:t,backend:n}=e,{inputIndices:r,inputShape:i,newShape:a}=t;if(r.shape.length!==2)throw Error(`Input indices should be a matrix but received shape ${r.shape}`);if(i.shape.length!==1)throw Error(`Input shape should be a vector but received shape ${i.shape}`);if(a.shape.length!==1)throw Error(`Target shape should be a vector but received shape ${a.shape}`);let o=Array.from(n.readSync(i.dataId)),s=n.readSync(r.dataId),c=Array.from(n.readSync(a.dataId)),[l,u,d]=yD(s,r.shape,r.dtype,o,c);return[n.makeTensorInfo(u,r.dtype,l),n.makeTensorInfo([d.length],a.dtype,new Int32Array(d))]}var lF={kernelName:er,backendName:`webgl`,kernelFunc:cF};function uF(e){let{inputs:t,backend:n}=e,{data:r,indices:i,segmentIds:a}=t;if(r.shape.length<1)throw Error(`Data should be at least 1 dimensional but received scalar`);if(i.shape.length!==1)throw Error(`Indices should be a vector but received shape
              ${i.shape}`);if(a.shape.length!==1)throw Error(`Segment ids should be a vector but received shape
              ${a.shape}`);let o=n.readSync(r.dataId),s=n.readSync(i.dataId),c=n.readSync(a.dataId),[l,u]=bD(o,r.shape,r.dtype,s,c,!0);return n.makeTensorInfo(u,r.dtype,l)}var dF={kernelName:tr,backendName:`webgl`,kernelFunc:uF};function fF(e){let{inputs:t,backend:n}=e,{data:r,indices:i,segmentIds:a}=t;if(r.shape.length<1)throw Error(`Data should be at least 1 dimensional but received scalar`);if(i.shape.length!==1)throw Error(`Indices should be a vector but received shape
             ${i.shape}`);if(a.shape.length!==1)throw Error(`Segment ids should be a vector but received shape
             ${a.shape}`);let o=n.readSync(r.dataId),s=n.readSync(i.dataId),c=n.readSync(a.dataId),[l,u]=bD(o,r.shape,r.dtype,s,c);return n.makeTensorInfo(u,r.dtype,l)}var pF={kernelName:nr,backendName:`webgl`,kernelFunc:fF};function mF(e){let{inputs:t,backend:n,attrs:r}=e,{sparseIndices:i,sparseValues:a,defaultValue:o}=t,{outputShape:s}=r,{sliceRank:c,numUpdates:l,sliceSize:u,strides:d,outputSize:f}=$p(a,i,s);if(a.dtype===`string`){let e=mD(n.bufferSync(i),n.bufferSync(a),s,f,u,l,c,d,ii(n.readSync(o.dataId)[0]),!1);return n.makeTensorInfo(s,e.dtype,e.values)}let p=new UP(l,c,i.shape.length,a.shape.length,d,[f,1],!1),m=n.runWebGLProgram(p,[a,i,o],a.dtype),h=$({inputs:{x:m},backend:n,attrs:{shape:s}});return n.disposeIntermediateTensorInfo(m),h}var hF={kernelName:rr,backendName:`webgl`,kernelFunc:mF};function gF(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{numOrSizeSplits:a,axis:o}=r,s=D(o,i.shape)[0],c=Qv(i,a,s),l=i.shape.length,u=Array(l).fill(0),d=i.shape.slice();return c.map(e=>{let t=[...d];t[s]=e;let r=rA({inputs:{x:i},backend:n,attrs:{begin:u,size:t}});return u[s]+=e,r})}var _F={kernelName:Zn,backendName:`webgl`,kernelFunc:gF},vF=`return sqrt(x);`,yF={kernelName:Yn,backendName:`webgl`,kernelFunc:Q({opSnippet:vF,packedOpSnippet:vF,cpuKernelImpl:xD})},bF={kernelName:ar,backendName:`webgl`,kernelFunc:Q({opSnippet:`return x * x;`})},xF=`return (a - b) * (a - b);`,SF={kernelName:ir,backendName:`webgl`,kernelFunc:PO({opSnippet:xF,packedOpSnippet:xF})};function CF(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t;if(i.dtype!==`string`)throw Error(`Input must be of datatype string`);let a=SD(gy(n.readSync(i.dataId)),`string`,r);return n.makeTensorInfo(i.shape,`string`,a)}var wF={kernelName:or,backendName:`webgl`,kernelFunc:CF};function TF({inputs:e,attrs:t,backend:n}){let{x:r}=e,i=KD+`
    return x > 0.0 ? 1.0 : float(${t.alpha});
  `,a=new GD(r.shape,i);return n.runWebGLProgram(a,[r],r.dtype)}var EF={kernelName:br,backendName:`webgl`,kernelFunc:TF},DF=class{constructor(e,t,n){this.variableNames=[`x`],this.outputShape=n;let r=n.length,i=Z(n.length),a=Z(n.length),o=``;if(r===1)o=`coords * strides + begin`;else{let e=0;o=n.map((t,r)=>(e++,n.length===1?`coords * strides[${r}] + begin[${r}]`:`coords[${e-1}] * strides[${r}] + begin[${r}]`)).join(`,`)}this.userCode=`
      ${i} begin = ${i}(${e});
      ${i} strides = ${i}(${t});

      void main() {
        ${a} coords = getOutputCoords();
        setOutput(getX(${o}));
      }
    `}};function OF(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{begin:a,end:o,strides:s,beginMask:c,endMask:l,ellipsisMask:u,newAxisMask:d,shrinkAxisMask:f}=r,{finalShapeSparse:p,finalShape:m,isIdentity:h,sliceDim0:g,isSimpleSlice:_,begin:v,end:b,strides:x}=sv(i.shape,a,o,s,c,l,u,d,f),S;if(h)S=$({inputs:{x:i},backend:n,attrs:{shape:m}});else if(g||_){y(i.shape.length>=1,()=>`Input must have rank at least 1, got: ${i.shape.length}`);let e=J_(v,b,x),t=rA({inputs:{x:i},backend:n,attrs:{begin:v,size:e}});S=$({inputs:{x:t},backend:n,attrs:{shape:m}}),n.disposeIntermediateTensorInfo(t)}else if(n.shouldExecuteOnCPU([i])){let e=n.readSync(i.dataId),t=CD(p,I(i.shape,i.dtype,e),x,v);S=n.makeTensorInfo(m,i.dtype,t.values)}else{let e=new DF(v,x,p);S=n.runWebGLProgram(e,[i],i.dtype)}let C=$({inputs:{x:S},backend:n,attrs:{shape:m}});return n.disposeIntermediateTensorInfo(S),C}var kF={kernelName:sr,backendName:`webgl`,kernelFunc:OF};function AF(e){let{inputs:t,backend:n,attrs:r}=e,{separator:i,nGramWidths:a,leftPad:o,rightPad:s,padWidth:c,preserveShortSequences:l}=r,{data:u,dataSplits:d}=t,[f,p]=wD(n.readSync(u.dataId),n.readSync(d.dataId),i,a,o,s,c,l);return[n.makeTensorInfo([f.length],`string`,f),n.makeTensorInfo(d.shape,`int32`,p)]}var jF={kernelName:cr,backendName:`webgl`,kernelFunc:AF};function MF(e){let{inputs:t,backend:n,attrs:r}=e,{skipEmpty:i}=r,{input:a,delimiter:o}=t;if(a.dtype!==`string`)throw Error(`Input must be of datatype string`);if(a.shape.length!==1)throw Error(`Input must be a vector, got shape: ${a.shape}`);if(o.shape.length!==0)throw Error(`Delimiter must be a scalar, got shape: ${o.shape}`);let s=n.readSync(a.dataId),c=n.readSync(o.dataId)[0],[l,u,d]=TD(s,c,i),f=u.length;return[n.makeTensorInfo([f,2],`int32`,l),n.makeTensorInfo([f],`string`,u),n.makeTensorInfo([2],`int32`,new Int32Array(d))]}var NF={kernelName:lr,backendName:`webgl`,kernelFunc:MF};function PF(e){let{inputs:t,backend:n,attrs:r}=e,{numBuckets:i}=r,{input:a}=t;if(a.dtype!==`string`)throw Error(`Input must be of datatype string`);if(i<=0)throw Error(`Number of buckets must be at least 1`);let o=ED(n.readSync(a.dataId),i);return n.makeTensorInfo(a.shape,`int32`,o)}var FF={kernelName:ur,backendName:`webgl`,kernelFunc:PF},IF={kernelName:`Tan`,backendName:`webgl`,kernelFunc:Q({opSnippet:`return tan(x);`})},LF={kernelName:dr,backendName:`webgl`,kernelFunc:Q({opSnippet:`
  float e2x = exp(-2.0 * abs(x));
  return sign(x) * (1.0 - e2x) / (1.0 + e2x);
`})};function RF(e){let{inputs:t,backend:n,attrs:r}=e,{tensor:i,indices:a,updates:o}=t,{}=r,{sliceRank:s,numUpdates:c,sliceSize:l,strides:u,outputSize:d}=$p(o,a,i.shape),f=[d/l,l];if(d===0)return n.makeTensorInfo(i.shape,a.dtype);let p=$({inputs:{x:a},backend:n,attrs:{shape:[c,s]}}),m=$({inputs:{x:o},backend:n,attrs:{shape:[c,l]}}),h=$({inputs:{x:i},backend:n,attrs:{shape:f}}),g=new UP(c,s,p.shape.length,m.shape.length,u,f,!1,!0),_=n.runWebGLProgram(g,[m,p,h],h.dtype),v=$({inputs:{x:_},backend:n,attrs:{shape:i.shape}});return n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(h),n.disposeIntermediateTensorInfo(_),v}var zF={kernelName:Bn,backendName:`webgl`,kernelFunc:RF},BF=class{constructor(e,t){this.variableNames=[`A`];let n=Array(e.length);for(let r=0;r<n.length;r++)n[r]=e[r]*t[r];this.outputShape=n,this.rank=n.length;let r=Z(this.rank),i=VF(e);this.userCode=`
      void main() {
        ${r} resRC = getOutputCoords();
        setOutput(getA(${i}));
      }
    `}};function VF(e){let t=e.length;if(t>5)throw Error(`Tile for rank ${t} is not yet supported`);if(t===1)return`imod(resRC, ${e[0]})`;let n=[`resRC.x`,`resRC.y`,`resRC.z`,`resRC.w`,`resRC.u`],r=[];for(let t=0;t<e.length;t++)r.push(`imod(${n[t]}, ${e[t]})`);return r.join()}function HF(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{reps:a}=r;if(i.dtype===`string`||i.shape.length>5){let e=n.readSync(i.dataId),t=i.dtype===`string`?e.map(e=>ii(e)):e,r=OD(I(i.shape,i.dtype,t),a);return n.makeTensorInfo(r.shape,r.dtype,r.values)}let o=new BF(i.shape,a);return n.runWebGLProgram(o,[i],i.dtype)}var UF={kernelName:fr,backendName:`webgl`,kernelFunc:HF},WF=class{constructor(e){this.variableNames=[`x`,`indices`],this.customUniforms=[{name:`n`,type:`int`},{name:`firstPass`,type:`int`},{name:`negativeInf`,type:`float`},{name:`dir`,type:`int`},{name:`inc`,type:`int`}],this.outputShape=e,this.userCode=`
       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // We compare elements pair-wise within a group of size 2 * inc.
         // The comparing rule for each group alternates between ascending
         // and descending. Within each group, we compare each pair at
         // positions i and i+inc. To decide whether an element at position i
         // is x0 or x1, we mod it by 2 * inc, if the result is smaller than
         // inc, it is in the first half of the group, we denote it as x0,
         // otherwise we denote it as x1.
         // For example, as shown in the Bitonic top K paper referenced above,
         // Figure5(a) shows that element[1] is in the
         // second half of the group when group size is 2, but it is in the
         // first half of the group when group size is 4.

         bool isFirstInPair = imod(elemIdx, 2 * inc) < inc;
         int i = isFirstInPair ? elemIdx : elemIdx - inc;

         int i0 = firstPass == 1 ? i : int(getIndices(batch, i));
         int i1 = firstPass == 1 ? i + inc : int(getIndices(batch, i + inc));
         float x0 = i0 < n ? getX(batch, i0) : negativeInf;
         float x1 = i1 < n ? getX(batch, i1) : negativeInf;

         // Denotes which direction indices are in (ascending or descending).
         bool reverse = imod(elemIdx, 2 * dir) >= dir;
         bool isGreater = x0 > x1 || (x0 == x1 && i1 > i0);
         if (reverse == isGreater) { // Elements in opposite order of direction
           int iTemp = i0;
           i0 = i1;
           i1 = iTemp;
         }
         if (isFirstInPair) {
            setOutput(float(i0));
         } else {
            setOutput(float(i1));
         }
       }
     `}},GF=class{constructor(e){this.variableNames=[`x`,`indices`],this.customUniforms=[{name:`n`,type:`int`},{name:`firstPass`,type:`int`},{name:`k`,type:`int`}],this.outputShape=e,this.userCode=`
    void main() {
         // Takes max of indices (0, k), (1, k + 1), (2, k + 2) ...
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // The output size is half of the previous size.
         // If the previous sequence is | | | | _ _ _ _  | | | |  _ _ _ _ (k=4),
         // we only need to output the indices at positions |, the indices at
         // positions _ can be thrown away, see Figure5(b) After Phase 2
         // (Merge phase) in the Bitonic Top K paper referenced above.
         // For example, the paper shows we only need to output the orange bars.
         // The output sequence should look like this | | | | | | | |.
         // Because the sequence is halved, to map the output index back
         // to the previous sequence to find the corresponding value,
         // we need to double the index. When we double the index,
         // we basically interpolate a position, so 2i looks like
         // | _ | _ | _ | _ | _ | _ | _. We move the | to the first k position
         // of each 2k positions by - elemIdx % k. E.g. for output at
         // index 4,5,6,7, we want to get the corresponding element at
         // original index 8,9,10,11, for output at index 8,9,10,11,
         // we want to get the corresponding element at original index
         // 16,17,18,19, so on and so forth.

         int i = elemIdx < k ? elemIdx : (elemIdx * 2 - imod(elemIdx, k));
         int i0 = firstPass == 1 ? i : int(getIndices(batch, i));
         int i1 = firstPass == 1 ? i + k : int(getIndices(batch, i + k));

         float x0 = getX(batch, i0);
         float x1 = i1 < n ? getX(batch, i1) : x0;

         setOutput(x0 >= x1 ? float(i0) : float(i1));
       }
     `}};function KF(e,t){t!==null&&e.disposeIntermediateTensorInfo(t)}function qF(e){let t=1;for(;t<e;)t*=2;return t}function JF(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{k:a,sorted:o}=r,s=O().getNumber(`TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD`),c=O().getNumber(`TOPK_K_CPU_HANDOFF_THRESHOLD`),l=i.shape,u=l[l.length-1];if(n.shouldExecuteOnCPU([i])||u<s||a>c){let[e,t]=kD(n.readSync(i.dataId),l,i.dtype,a,o);return[n.makeTensorInfo(e.shape,e.dtype,e.values),n.makeTensorInfo(t.shape,t.dtype,t.values)]}if(a===0)return l[l.length-1]=0,[n.makeTensorInfo(l,i.dtype,[]),n.makeTensorInfo(l,`int32`,[])];if(u===1)return[i,uM({attrs:{shape:l,dtype:`int32`,value:0},backend:n})];let d=n.texData.get(i.dataId),f=d!==null&&d.isPacked,p=f?n.unpackTensor(i):i,m=S(l)/u,h=$({inputs:{x:p},attrs:{shape:[m,u]},backend:n});f&&KF(n,p);let g=qF(a),_=qF(u),v=null,y=()=>v===null?[h,h]:[h,v],b=(e,t,r)=>{let i=y(),a=new WF(r),o=[[u],[+(v===null)],[-1/0],[e],[t]],s=v;v=n.runWebGLProgram(a,i,`int32`,o),KF(n,s)};for(let e=1;e<g;e*=2){let t=e*2;for(let n=e;n>=1;n/=2)b(t,n,[m,_])}for(let e=_;e>g;e/=2){let t=y(),r=new GF([m,e/2]),i=[[u],[+(v===null)],[g]],a=v;v=n.runWebGLProgram(r,t,`int32`,i),KF(n,a);let o=g/2,s=o*2;for(let e=o;e>=1;e/=2)b(s,e,v.shape)}let x=v;v=rA({inputs:{x:v},backend:n,attrs:{begin:0,size:[m,a]}}),KF(n,x);let C=MM({inputs:{x:h,indices:v},backend:n,attrs:{axis:1,batchDims:1}});KF(n,h);let w=l.slice(0,-1);w.push(a),x=v,v=$({inputs:{x:v},attrs:{shape:w},backend:n}),KF(n,x);let ee=C;return C=$({inputs:{x:C},attrs:{shape:w},backend:n}),KF(n,ee),[C,v]}var YF={kernelName:pr,backendName:`webgl`,kernelFunc:JF},XF=class{constructor(e,t,n,r,i,a){this.variableNames=[`Image`,`Transforms`],this.outputShape=a;let o=n===`nearest`?1:2,s;switch(r){case`constant`:s=1;break;case`reflect`:s=2;break;case`wrap`:s=3;break;case`nearest`:s=4;break;default:s=1;break}this.userCode=`
            float mapCoord(float outCoord, float len) {
              float inCoord = outCoord;
              if(${s} == 2) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    if (inCoord < sz2) {
                      inCoord = sz2 * float(int(float(-inCoord / sz2))) +
                      inCoord;
                    }
                    inCoord = inCoord < -len ? inCoord + sz2 : -inCoord - 1.0;
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    inCoord -= sz2 * float(int(float(inCoord / sz2)));
                    if (inCoord >= len) {
                      inCoord = sz2 - inCoord - 1.0;
                    }
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (${s} == 3) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord += len * (float(int(float(-inCoord / sz))) + 1.0);
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord -= len * float(int(float(inCoord / sz)));
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (${s} == 4) {
                return clamp(outCoord, 0.0, len - 1.0);
              } else {
                return outCoord;
              }
            }

            float readWithFillValue(int batch, int coordY, int coordX,
              int channel) {
              float outputValue;
              if (0 <= coordY && coordY < ${e} && 0 <= coordX && coordX < ${t}) {
                  outputValue = getImage(batch, coordY, coordX, channel);
              } else {
                outputValue = float(${i});
              }
              return outputValue;
            }

            void main() {
              ivec4 coords = getOutputCoords();
              float outputValue;
              int batch = coords[0];
              int x = coords[2];
              int y = coords[1];
              int channel = coords[3];
              float xf = float(x);
              float yf = float(y);
              float a1 = getTransforms(batch, 0);
              float a2 = getTransforms(batch, 1);
              float a3 = getTransforms(batch, 2);
              float b1 = getTransforms(batch, 3);
              float b2 = getTransforms(batch, 4);
              float b3 = getTransforms(batch, 5);
              float c1 = getTransforms(batch, 6);
              float c2 = getTransforms(batch, 7);
              float projection = c1 * xf + c2 * yf + 1.0;
              if (projection == 0.0) {
                outputValue = float(${i});
              } else {
                float inX = (a1 * xf + a2 * yf + a3) / projection;
                float inY = (b1 * xf + b2 * yf + b3) / projection;
                float mapX = mapCoord(inX, float(${t}));
                float mapY = mapCoord(inY, float(${e}));

                if (${o} == 1) {
                  int coordY = int(round(mapY));
                  int coordX = int(round(mapX));
                  outputValue = readWithFillValue(batch, coordY, coordX,
                    channel);
                } else {
                  float yFloor = floor(mapY);
                  float xFloor = floor(mapX);
                  float yCeil = yFloor + 1.0;
                  float xCeil = xFloor + 1.0;
                  float valueYFloor = (xCeil - mapX) *
                  readWithFillValue(batch, int(yFloor), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yFloor), int(xCeil), channel);
                  float valueYCeil = (xCeil - mapX) *
                  readWithFillValue(batch, int(yCeil), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yCeil), int(xCeil), channel);
                  outputValue = (yCeil - mapY) * valueYFloor +
                  (mapY - yFloor) * valueYCeil;
                }
              }
              setOutput(outputValue);
            }
        `}};function ZF(e){let{inputs:t,backend:n,attrs:r}=e,{image:i,transforms:a}=t,{interpolation:o,fillMode:s,fillValue:c,outputShape:l}=r,[u,d,f,p]=i.shape,[m,h]=l??[d,f],g=new XF(d,f,o,s,c,[u,m,h,p]);return n.runWebGLProgram(g,[i,a],`float32`)}var QF={kernelName:mr,backendName:`webgl`,kernelFunc:ZF};function $F(e){let{inputs:t,attrs:n,backend:r}=e,{axis:i}=n,{x:a}=t;jC(a,`unique`),console.warn(`WARNING: `,`UI might be locked temporarily as data is being downloaded`);let{outputValues:o,outputShape:s,indices:c}=jD(r.readSync(a.dataId),i,a.shape,a.dtype);return[r.makeTensorInfo(s,a.dtype,o),r.makeTensorInfo([c.length],`int32`,c)]}var eI={kernelName:gr,backendName:`webgl`,kernelFunc:$F};function tI(e){let{inputs:t,backend:n,attrs:r}=e,{value:i}=t,{axis:a}=r;a<0&&(a+=i.shape.length);let o=i,s=o.shape.length,c=i.shape[a],l=Array(s-1),u=0;for(let e=0;e<s;e++)e!==a&&(l[u++]=o.shape[e]);let d=[],f=Array(s).fill(0),p=o.shape.slice();p[a]=1;let m=Array(c);for(let e=0;e<m.length;e++){f[a]=e;let t=rA({inputs:{x:o},backend:n,attrs:{begin:f,size:p}});m[e]=$({inputs:{x:t},backend:n,attrs:{shape:l}}),d.push(t)}return d.forEach(e=>n.disposeIntermediateTensorInfo(e)),m}var nI={kernelName:_r,backendName:`webgl`,kernelFunc:tI},rI=class{constructor(e,t){this.variableNames=[`x`,`segmentIds`];let n=e.windowSize,r=e.batchSize,i=e.inSize,a=e.numSegments,o=a*Math.ceil(i/n);this.outputShape=[r,o];let s=Math.floor(n/4)*4,c=n%4,l=`
        sumValue += dot(values, segFilter);
    `,u=``;i%n>0&&(u=`
        if (inIdx < 0 || inIdx >= ${i}) {
          return initializationValue;
        }
      `);let d=``;i%n>0&&(d=`
        if (inIdx < 0 || inIdx >= ${i}) {
          return -1.0;
        }
      `),this.userCode=`
      const float initializationValue = 0.0;

      float getValue(int batch, int inIdx) {
        ${u}
        return getX(batch, inIdx);
      }

      float getSegmentIdAtIndex(int inIdx) {
        ${d}
        return getSegmentIds(inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = int(floor(float(outIdx) / float(
          ${a})) * float(${n}));
        int currentSeg = int(mod(float(outIdx), float(${a})));

        float sumValue = 0.0;

        for (int i = 0; i < ${s}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 3)) == currentSeg ? 1 : 0
          );

          ${l}
        }

        int inIdx = inOffset + ${s};
        if (${c===1}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          int inIdxSeg = int(getSegmentIdAtIndex(inIdx));

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            0,
            0,
            0
          );

          ${l}
        } else if (${c===2}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
              0,
              0
          );

          ${l}
        } else if (${c===3}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            0
          );

          ${l}
        }
        setOutput(sumValue);
      }
    `}};function iI(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,segmentIds:a}=t,{numSegments:o}=r,s=i.shape.length,c=[],l=0,u=Ol([l],s),d=i;u!=null&&(d=tk({inputs:{x:i},backend:n,attrs:{perm:u}}),c.push(d),l=Al(1,s)[0]);let f=py(d.shape,l,o),p=S([d.shape[l]]),m=$({inputs:{x:d},backend:n,attrs:{shape:[-1,p]}});c.push(m);let h=Ii(i.dtype),g=(e,t,r,i,a)=>{let o=e.shape[0],s=e.shape[1],l=fy(s,a),u=new rI({windowSize:l,inSize:s,batchSize:o,numSegments:a},t),d=n.compileAndRun(u,[e,r],i);if(c.push(d),d.shape[1]===a)return d;let f=gP({backend:n,attrs:{start:0,stop:a,step:1,dtype:`float32`}}),p=HF({inputs:{x:f},backend:n,attrs:{reps:[s/l]}});return c.push(f),c.push(p),g(d,t,p,i,a)},_=$({inputs:{x:g(m,`unsortedSegmentSum`,a,h,o)},backend:n,attrs:{shape:f}}),v=_;if(u!=null){c.push(_);let e=kl(u);v=tk({inputs:{x:v},backend:n,attrs:{perm:e}})}return c.forEach(e=>n.disposeIntermediateTensorInfo(e)),v}var aI=[ak,ck,lk,uk,fk,gk,vk,bk,Dk,kk,Ak,jk,Mk,Nk,Pk,Rk,Bk,Wk,Kk,Jk,Zk,aA,sA,dA,pA,xA,CA,DA,wO,jA,BA,YA,tj,ij,oj,cj,uj,dj,fj,mj,xj,Cj,Tj,Oj,Mj,Ij,Rj,Vj,Wj,Kj,qj,Xj,Zj,Qj,eM,nM,iM,cM,dM,pM,hM,gM,yM,wM,EM,kM,NM,PM,FM,SO,LM,IA,RM,zM,BM,OO,VM,HM,WM,GM,KM,qM,JM,YM,QM,eN,rN,iN,oN,cN,fN,mN,gN,vN,bN,xN,wN,TN,FN,VO,zN,HN,GN,JN,hA,XN,eP,nP,oP,sP,MO,lP,dP,pP,hP,_P,_A,ON,vP,yP,bP,UO,wP,DP,jP,PP,RP,BP,VP,HP,KP,YP,QP,$P,eF,tF,nF,rF,iA,NN,iF,aF,sF,lF,dF,pF,hF,_F,yF,bF,SF,wF,EF,kF,jF,NF,FF,jN,ek,IF,LF,zF,UF,YF,QF,nk,eI,nI,{kernelName:vr,backendName:`webgl`,kernelFunc:iI},QN];for(let e of aI)Nr(e);var oI=2,sI=.07,cI=class{detector=null;isRunning=!1;animFrameId=null;frameCallback=null;videoEl=null;config;constructor(e={}){this.config=e}async init(){let e=yS.MediaPipeHands,t={runtime:`tfjs`,modelType:this.config.lite?`lite`:`full`,maxHands:this.config.maxHands??oI};this.detector=await SS(e,t)}start(e){if(!this.detector)throw Error(`HandTracker not initialised. Call init() first.`);this.videoEl=e,this.isRunning=!0,this.loop()}stop(){this.isRunning=!1,this.animFrameId!==null&&cancelAnimationFrame(this.animFrameId),this.detector?.dispose(),this.detector=null}onFrame(e){this.frameCallback=e}loop(){!this.isRunning||!this.videoEl||!this.detector||this.detector.estimateHands(this.videoEl,{flipHorizontal:!1}).then(e=>{let t=e.map(e=>{let t=e.keypoints.map(e=>({x:e.x,y:e.y,z:e.z??0,...e.name===void 0?{}:{name:e.name}}));return{handedness:e.handedness,score:e.score??0,landmarks:t,gesture:lI(t)}});this.frameCallback?.(t),this.animFrameId=requestAnimationFrame(()=>this.loop())}).catch(e=>{console.warn(`[HandTracker] Frame error:`,e),this.animFrameId=requestAnimationFrame(()=>this.loop())})}};function lI(e){if(e.length<21)return`unknown`;let t=e[0],n=e[4],r=e[8],i=e[12],a=e[16],o=e[20],s=e[5],c=e[9],l=e[13],u=e[17],d=(e,n)=>uI(e,t)>uI(n,t)*1.2,f=d(r,s),p=d(i,c),m=d(a,l),h=d(o,u);return uI(n,r)<sI*uI(t,s)*10&&!p&&!m&&!h?`pinch`:f&&!p&&!m&&!h?`point`:!f&&!p&&!m&&!h?`fist`:f&&p&&m&&h?`open`:`unknown`}function uI(e,t){let n=e.x-t.x,r=e.y-t.y,i=e.z-t.z;return Math.sqrt(n*n+r*r+i*i)}export{cI as HandTracker};