(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const l of document.querySelectorAll('link[rel="modulepreload"]'))n(l);new MutationObserver(l=>{for(const i of l)if(i.type==="childList")for(const r of i.addedNodes)r.tagName==="LINK"&&r.rel==="modulepreload"&&n(r)}).observe(document,{childList:!0,subtree:!0});function s(l){const i={};return l.integrity&&(i.integrity=l.integrity),l.referrerPolicy&&(i.referrerPolicy=l.referrerPolicy),l.crossOrigin==="use-credentials"?i.credentials="include":l.crossOrigin==="anonymous"?i.credentials="omit":i.credentials="same-origin",i}function n(l){if(l.ep)return;l.ep=!0;const i=s(l);fetch(l.href,i)}})();const Q=(e,t)=>e===t,P={equals:Q};let j=q;const g=1,m=2,D={owned:null,cleanups:null,context:null,owner:null};var a=null;let L=null,f=null,u=null,d=null,E=0;function V(e,t){const s=f,n=a,l=e.length===0,i=t===void 0?n:t,r=l?D:{owned:null,cleanups:null,context:i?i.context:null,owner:i},o=l?e:()=>e(()=>v(()=>T(r)));a=r,f=null;try{return b(o,!0)}finally{f=s,a=n}}function W(e,t){t=t?Object.assign({},P,t):P;const s={value:e,observers:null,observerSlots:null,comparator:t.equals||void 0},n=l=>(typeof l=="function"&&(l=l(s.value)),M(s,l));return[X.bind(s),n]}function S(e,t,s){const n=R(e,t,!1,g);N(n)}function J(e,t,s){j=z;const n=R(e,t,!1,g);(!s||!s.render)&&(n.user=!0),d?d.push(n):N(n)}function v(e){if(f===null)return e();const t=f;f=null;try{return e()}finally{f=t}}function X(){if(this.sources&&this.state)if(this.state===g)N(this);else{const e=u;u=null,b(()=>x(this),!1),u=e}if(f){const e=this.observers?this.observers.length:0;f.sources?(f.sources.push(this),f.sourceSlots.push(e)):(f.sources=[this],f.sourceSlots=[e]),this.observers?(this.observers.push(f),this.observerSlots.push(f.sources.length-1)):(this.observers=[f],this.observerSlots=[f.sources.length-1])}return this.value}function M(e,t,s){let n=e.value;return(!e.comparator||!e.comparator(n,t))&&(e.value=t,e.observers&&e.observers.length&&b(()=>{for(let l=0;l<e.observers.length;l+=1){const i=e.observers[l],r=L&&L.running;r&&L.disposed.has(i),(r?!i.tState:!i.state)&&(i.pure?u.push(i):d.push(i),i.observers&&G(i)),r||(i.state=g)}if(u.length>1e6)throw u=[],new Error},!1)),t}function N(e){if(!e.fn)return;T(e);const t=a,s=f,n=E;f=a=e,Y(e,e.value,n),f=s,a=t}function Y(e,t,s){let n;try{n=e.fn(t)}catch(l){return e.pure&&(e.state=g,e.owned&&e.owned.forEach(T),e.owned=null),e.updatedAt=s+1,H(l)}(!e.updatedAt||e.updatedAt<=s)&&(e.updatedAt!=null&&"observers"in e?M(e,n):e.value=n,e.updatedAt=s)}function R(e,t,s,n=g,l){const i={fn:e,state:n,updatedAt:null,owned:null,sources:null,sourceSlots:null,cleanups:null,value:t,owner:a,context:a?a.context:null,pure:s};return a===null||a!==D&&(a.owned?a.owned.push(i):a.owned=[i]),i}function A(e){if(e.state===0)return;if(e.state===m)return x(e);if(e.suspense&&v(e.suspense.inFallback))return e.suspense.effects.push(e);const t=[e];for(;(e=e.owner)&&(!e.updatedAt||e.updatedAt<E);)e.state&&t.push(e);for(let s=t.length-1;s>=0;s--)if(e=t[s],e.state===g)N(e);else if(e.state===m){const n=u;u=null,b(()=>x(e,t[0]),!1),u=n}}function b(e,t){if(u)return e();let s=!1;t||(u=[]),d?s=!0:d=[],E++;try{const n=e();return Z(s),n}catch(n){s||(d=null),u=null,H(n)}}function Z(e){if(u&&(q(u),u=null),e)return;const t=d;d=null,t.length&&b(()=>j(t),!1)}function q(e){for(let t=0;t<e.length;t++)A(e[t])}function z(e){let t,s=0;for(t=0;t<e.length;t++){const n=e[t];n.user?e[s++]=n:A(n)}for(t=0;t<s;t++)A(e[t])}function x(e,t){e.state=0;for(let s=0;s<e.sources.length;s+=1){const n=e.sources[s];if(n.sources){const l=n.state;l===g?n!==t&&(!n.updatedAt||n.updatedAt<E)&&A(n):l===m&&x(n,t)}}}function G(e){for(let t=0;t<e.observers.length;t+=1){const s=e.observers[t];s.state||(s.state=m,s.pure?u.push(s):d.push(s),s.observers&&G(s))}}function T(e){let t;if(e.sources)for(;e.sources.length;){const s=e.sources.pop(),n=e.sourceSlots.pop(),l=s.observers;if(l&&l.length){const i=l.pop(),r=s.observerSlots.pop();n<l.length&&(i.sourceSlots[r]=n,l[n]=i,s.observerSlots[n]=r)}}if(e.owned){for(t=e.owned.length-1;t>=0;t--)T(e.owned[t]);e.owned=null}if(e.cleanups){for(t=e.cleanups.length-1;t>=0;t--)e.cleanups[t]();e.cleanups=null}e.state=0}function k(e){return e instanceof Error?e:new Error(typeof e=="string"?e:"Unknown error",{cause:e})}function H(e,t=a){throw k(e)}function _(e,t){return v(()=>e(t||{}))}function ee(e,t,s){let n=s.length,l=t.length,i=n,r=0,o=0,c=t[l-1].nextSibling,p=null;for(;r<l||o<i;){if(t[r]===s[o]){r++,o++;continue}for(;t[l-1]===s[i-1];)l--,i--;if(l===r){const h=i<n?o?s[o-1].nextSibling:s[i-o]:c;for(;o<i;)e.insertBefore(s[o++],h)}else if(i===o)for(;r<l;)(!p||!p.has(t[r]))&&t[r].remove(),r++;else if(t[r]===s[i-1]&&s[o]===t[l-1]){const h=t[--l].nextSibling;e.insertBefore(s[o++],t[r++].nextSibling),e.insertBefore(s[--i],h),t[l]=s[i]}else{if(!p){p=new Map;let y=o;for(;y<i;)p.set(s[y],y++)}const h=p.get(t[r]);if(h!=null)if(o<h&&h<i){let y=r,$=1,F;for(;++y<l&&y<i&&!((F=p.get(t[y]))==null||F!==h+$);)$++;if($>h-o){const K=t[r];for(;o<h;)e.insertBefore(s[o++],K)}else e.replaceChild(s[o++],t[r++])}else r++;else t[r++].remove()}}}function te(e,t,s,n={}){let l;return V(i=>{l=i,t===document?e():B(t,e(),t.firstChild?null:void 0,s)},n.owner),()=>{l(),t.textContent=""}}function U(e,t,s){let n;const l=()=>{const r=document.createElement("template");return r.innerHTML=e,s?r.content.firstChild.firstChild:r.content.firstChild},i=t?()=>v(()=>document.importNode(n||(n=l()),!0)):()=>(n||(n=l())).cloneNode(!0);return i.cloneNode=i,i}function se(e,t){t==null?e.removeAttribute("class"):e.className=t}function B(e,t,s,n){if(s!==void 0&&!n&&(n=[]),typeof t!="function")return C(e,t,n,s);S(l=>C(e,t(),l,s),n)}function C(e,t,s,n,l){for(;typeof s=="function";)s=s();if(t===s)return s;const i=typeof t,r=n!==void 0;if(e=r&&s[0]&&s[0].parentNode||e,i==="string"||i==="number")if(i==="number"&&(t=t.toString()),r){let o=s[0];o&&o.nodeType===3?o.data=t:o=document.createTextNode(t),s=w(e,s,n,o)}else s!==""&&typeof s=="string"?s=e.firstChild.data=t:s=e.textContent=t;else if(t==null||i==="boolean")s=w(e,s,n);else{if(i==="function")return S(()=>{let o=t();for(;typeof o=="function";)o=o();s=C(e,o,s,n)}),()=>s;if(Array.isArray(t)){const o=[],c=s&&Array.isArray(s);if(O(o,t,s,l))return S(()=>s=C(e,o,s,n,!0)),()=>s;if(o.length===0){if(s=w(e,s,n),r)return s}else c?s.length===0?I(e,o,n):ee(e,s,o):(s&&w(e),I(e,o));s=o}else if(t.nodeType){if(Array.isArray(s)){if(r)return s=w(e,s,n,t);w(e,s,null,t)}else s==null||s===""||!e.firstChild?e.appendChild(t):e.replaceChild(t,e.firstChild);s=t}else console.warn("Unrecognized value. Skipped inserting",t)}return s}function O(e,t,s,n){let l=!1;for(let i=0,r=t.length;i<r;i++){let o=t[i],c=s&&s[i],p;if(!(o==null||o===!0||o===!1))if((p=typeof o)=="object"&&o.nodeType)e.push(o);else if(Array.isArray(o))l=O(e,o,c)||l;else if(p==="function")if(n){for(;typeof o=="function";)o=o();l=O(e,Array.isArray(o)?o:[o],Array.isArray(c)?c:[c])||l}else e.push(o),l=!0;else{const h=String(o);c&&c.nodeType===3&&c.data===h?e.push(c):e.push(document.createTextNode(h))}}return l}function I(e,t,s=null){for(let n=0,l=t.length;n<l;n++)e.insertBefore(t[n],s)}function w(e,t,s,n){if(s===void 0)return e.textContent="";const l=n||document.createTextNode("");if(t.length){let i=!1;for(let r=t.length-1;r>=0;r--){const o=t[r];if(l!==o){const c=o.parentNode===e;!i&&!r?c?e.replaceChild(l,o):e.insertBefore(l,s):c&&o.remove()}else i=!0}}else e.insertBefore(l,s);return[l]}const ne={},le=U('<nav class="w-full flex justify-center h-[75px] items-center bg-sky-600"><h2 class="text-2xl text-white">Classification de genre musical');function ie(){return le()}const oe=U('<div class="relative inline-block"><label for="file-input" class="px-4 py-2 bg-blue-500 text-white rounded cursor-pointer hover:bg-blue-600">Sélectionner un fichier</label><input id="file-input" type="file" class="hidden">'),[he,re]=W();function fe(e){const t=s=>{re(s.target.files[0])};return J(()=>{console.log()}),(()=>{const s=oe();return s.firstChild.nextSibling.addEventListener("change",t),s})()}const ue=U('<div><section id="layout">'),ce=()=>(()=>{const e=ue(),t=e.firstChild;return B(e,_(ie,{}),t),B(t,_(fe,{onSelect:s=>{console.log("test",s)}})),S(()=>se(e,ne.App)),e})(),ae=document.getElementById("root");te(()=>_(ce,{}),ae);