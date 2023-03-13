//! This crate declares only the proc macro attributes, as a crate defining proc macro attributes
//! must not contain any other public items.
use syn::parse_macro_input;

use proc_macro::{self, TokenStream};
use quote::quote;
#[macro_use]
extern crate lazy_static;

/// A proc macro used to expose ciphercore Rust structs as Python objects.
#[proc_macro_attribute]
pub fn struct_wrapper(_metadata: TokenStream, input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as syn::ItemStruct);
    let expanded = macro_backend::build_struct(&ast.ident, &ast.attrs)
        .unwrap_or_else(|e| e.to_compile_error());
    quote!(#ast
        #expanded
    )
    .into()
}

/// A proc macro used to expose ciphercore Rust enums as Python objects.
#[proc_macro_attribute]
pub fn enum_to_struct_wrapper(_metadata: TokenStream, input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as syn::ItemEnum);
    let expanded = macro_backend::build_struct(&ast.ident, &ast.attrs)
        .unwrap_or_else(|e| e.to_compile_error());
    quote!(#ast
        #expanded
    )
    .into()
}

/// A proc macro used to expose methods to Python.
///
/// This self type must have one of {struct_wrapper, enum_to_struct_wrapper} attributes.
#[proc_macro_attribute]
pub fn impl_wrapper(_metadata: TokenStream, input: TokenStream) -> TokenStream {
    let mut ast = parse_macro_input!(input as syn::ItemImpl);
    let expanded = macro_backend::build_methods(&mut ast).unwrap_or_else(|e| e.to_compile_error());
    quote!(#ast
           #[allow(clippy::needless_question_mark)] #expanded
    )
    .into()
}

/// A proc macro used to expose single function to Python.
#[proc_macro_attribute]
pub fn fn_wrapper(_metadata: TokenStream, input: TokenStream) -> TokenStream {
    let mut ast = parse_macro_input!(input as syn::ItemFn);
    let expanded = macro_backend::build_fn(&mut ast).unwrap_or_else(|e| e.to_compile_error());
    quote!(#ast
           #expanded
    )
    .into()
}

mod macro_backend {
    use std::collections::HashSet;

    use proc_macro2::{Ident, TokenStream};
    use quote::{quote, ToTokens};
    use syn::{punctuated::Punctuated, token::Comma, FnArg, PatType, ReturnType};

    lazy_static! {
        static ref TYPES_TO_WRAP: HashSet<&'static str> = {
            HashSet::from_iter(vec![
                "Node",
                "Graph",
                "Context",
                "ScalarType",
                "Type",
                "SliceElement",
                "TypedValue",
                "Value",
                "CustomOperation",
                "JoinType",
                "ShardConfig",
            ])
        };
    }

    pub fn build_methods(ast: &mut syn::ItemImpl) -> syn::Result<TokenStream> {
        impl_methods(&ast.self_ty, &mut ast.items)
    }

    pub fn build_fn(ast: &mut syn::ItemFn) -> syn::Result<TokenStream> {
        let token_stream = gen_wrapper_method(&mut ast.sig, None)?;
        let attrs = gen_attributes(&ast.attrs);
        let name = ast.sig.ident.to_string();
        Ok(quote!(#(#attrs)*
    #[pyo3::pyfunction]
    #[pyo3(name = #name)]
    #token_stream))
    }

    pub fn build_struct(
        t: &syn::Ident,
        struct_attrs: &[syn::Attribute],
    ) -> syn::Result<TokenStream> {
        let nt = get_wrapper_ident(t);
        let name = format!("{}", t);
        let attrs = gen_attributes(struct_attrs);
        Ok(quote!(
            #(#attrs)*
            #[pyo3::pyclass(name = #name)]
            pub struct #nt {
                pub inner: #t,
            }
        ))
    }

    fn impl_methods(ty: &syn::Type, impls: &mut [syn::ImplItem]) -> syn::Result<TokenStream> {
        let mut methods = Vec::new();

        for iimpl in impls.iter_mut() {
            if let syn::ImplItem::Method(meth) = iimpl {
                let token_stream = gen_wrapper_method(&mut meth.sig, Some(ty))?;
                let attrs = gen_attributes(&meth.attrs);
                methods.push(quote!(#(#attrs)* #token_stream));
            }
        }
        let nt = get_wrapper_type_ident(ty, false);
        Ok(quote! {
            #[pyo3::pymethods]
            impl #nt {
            #(#methods)*
            fn __str__(&self) -> String {
                format!("{}", self.inner)
            }
            fn __repr__(&self) -> String {
                self.__str__()
            }
            }
        })
    }

    fn in_types_to_wrap(tt: &syn::Ident) -> bool {
        let ii = format!("{}", tt);
        TYPES_TO_WRAP.contains(ii.as_str())
    }

    fn get_wrapper_ident(tt: &syn::Ident) -> syn::Ident {
        let prefix = if in_types_to_wrap(tt) {
            "PyBinding"
        } else {
            ""
        };
        Ident::new(format!("{}{}", prefix, tt).as_str(), tt.span())
    }

    fn get_wrapper_type_ident(ty: &syn::Type, add_ref: bool) -> TokenStream {
        match get_last_path_segment_from_type_path(ty) {
            Some(s) => {
                let ident = get_wrapper_ident(&s.ident);
                if add_ref && in_types_to_wrap(&s.ident) {
                    quote!(&#ident)
                } else {
                    ident.to_token_stream()
                }
            }
            None => ty.to_token_stream(),
        }
    }

    fn gen_wrapper_method(
        sig: &mut syn::Signature,
        class: Option<&syn::Type>,
    ) -> syn::Result<TokenStream> {
        let name = &sig.ident;
        if let Some(ts) = check_in_allowlist(format!("{}", name)) {
            return Ok(ts);
        }
        let input = Input::new(&sig.inputs);
        let inner_inputs = input.get_inner_inputs();
        let sig_inputs = input.get_sig_inputs();
        let output = Output::new(&sig.output, class);
        let ret = output.get_output();
        let result = if class.is_some() {
            if input.has_receiver {
                output.wrap_result(quote!(self.inner.#name(#inner_inputs)))
            } else {
                let ts = class.to_token_stream();
                output.wrap_result(quote!(#ts::#name(#inner_inputs)))
            }
        } else {
            output.wrap_result(quote!(#name(#inner_inputs)))
        };
        let attr_sign = input.gen_attr_signature();
        let staticmethod = input.mb_gen_staticmethod(class.is_none());
        let prefix = if class.is_none() { "py_binding_" } else { "" };
        let result_fn_name = Ident::new(format!("{}{}", prefix, name).as_str(), sig.ident.span());
        Ok(quote!(#staticmethod #attr_sign pub fn #result_fn_name(#sig_inputs) #ret { #result }))
    }

    struct Output<'a> {
        has_result: bool,
        is_vector: bool,
        inner_type: Option<&'a Ident>,
        initial_return: &'a ReturnType,
    }

    impl<'a> Output<'a> {
        fn new(output: &'a ReturnType, class: Option<&'a syn::Type>) -> Self {
            let mut has_result = false;
            let mut is_vector = false;
            match &output {
                ReturnType::Default => Output {
                    has_result,
                    is_vector,
                    inner_type: None,
                    initial_return: output,
                },
                ReturnType::Type(_, t) => {
                    let s = match get_last_path_segment_from_type_path(t.as_ref()) {
                        Some(tt) => tt,
                        None => {
                            return Output {
                                has_result,
                                is_vector,
                                inner_type: None,
                                initial_return: output,
                            };
                        }
                    };
                    let ps = if format!("{}", s.ident) == "Result" {
                        has_result = true;
                        get_last_path_segment_from_first_argument(s)
                    } else {
                        Some(s)
                    };
                    let inner_type = match ps {
                        Some(p) => {
                            if format!("{}", p.ident) == "Vec" {
                                is_vector = true;
                                &get_last_path_segment_from_first_argument(p).unwrap().ident
                            } else if format!("{}", p.ident) == "Self" {
                                match get_last_path_segment_from_type_path(class.unwrap()) {
                                    Some(s) => &s.ident,
                                    None => &p.ident,
                                }
                            } else {
                                &p.ident
                            }
                        }
                        None => {
                            return Output {
                                has_result,
                                is_vector,
                                inner_type: None,
                                initial_return: output,
                            };
                        }
                    };
                    Output {
                        has_result,
                        is_vector,
                        inner_type: if in_types_to_wrap(inner_type) {
                            Some(inner_type)
                        } else {
                            None
                        },
                        initial_return: output,
                    }
                }
            }
        }
        fn get_output(&self) -> TokenStream {
            match self.inner_type {
                Some(t) => {
                    let name = get_wrapper_ident(t);
                    let mb_vec = if self.is_vector {
                        quote!(Vec<#name>)
                    } else {
                        name.to_token_stream()
                    };
                    if self.has_result {
                        quote!(-> pyo3::PyResult<#mb_vec>)
                    } else {
                        quote!(-> #mb_vec)
                    }
                }
                None => self.initial_return.to_token_stream(),
            }
        }
        fn wrap_result(&self, result: TokenStream) -> TokenStream {
            let return_if = if self.has_result {
                quote!(#result?)
            } else {
                result
            };
            let wrapped = if self.is_vector {
                match self.inner_type {
                    Some(t) => {
                        let name = get_wrapper_ident(t);
                        quote!(#return_if.into_iter().map(|x| #name {inner: x}).collect())
                    }
                    None => return_if,
                }
            } else {
                match self.inner_type {
                    Some(t) => {
                        let name = get_wrapper_ident(t);
                        quote!(#name {inner: #return_if})
                    }
                    None => return_if,
                }
            };

            if self.has_result {
                quote!(Ok(#wrapped))
            } else {
                wrapped
            }
        }
    }

    fn get_last_path_segment_from_first_argument(
        s: &syn::PathSegment,
    ) -> Option<&syn::PathSegment> {
        match &s.arguments {
            syn::PathArguments::AngleBracketed(args) => match args.args.first().unwrap() {
                syn::GenericArgument::Type(t) => match get_last_path_segment_from_type_path(t) {
                    Some(p) => Some(p),
                    None => None,
                },
                _ => None,
            },
            _ => None,
        }
    }

    struct InputArgument<'a> {
        is_vector: bool,
        initial_type: &'a syn::Type,
        inner_type: Option<&'a Ident>,
        var_name: TokenStream,
    }

    fn get_last_path_segment_from_type_path(t: &syn::Type) -> Option<&syn::PathSegment> {
        match t {
            syn::Type::Path(p) => match p.path.segments.last() {
                Some(s) => Some(s),
                None => None,
            },
            _ => None,
        }
    }

    impl<'a> InputArgument<'a> {
        fn new(t: &'a PatType) -> Self {
            let name = match t.pat.as_ref() {
                syn::Pat::Ident(i) => &i.ident,
                _ => unreachable!(),
            };
            let mut is_vector = false;
            let s = match get_last_path_segment_from_type_path(t.ty.as_ref()) {
                Some(s) => s,
                None => {
                    return InputArgument {
                        is_vector,
                        initial_type: &t.ty,
                        inner_type: None,
                        var_name: name.to_token_stream(),
                    }
                }
            };
            let inner_type = if format!("{}", s.ident) == "Vec" {
                is_vector = true;
                &get_last_path_segment_from_first_argument(s).unwrap().ident
            } else if format!("{}", s.ident) == "Slice" {
                is_vector = true;
                &s.ident
            } else {
                &s.ident
            };
            InputArgument {
                is_vector,
                initial_type: &t.ty,
                inner_type: if in_types_to_wrap(inner_type) || format!("{}", inner_type) == "Slice"
                {
                    Some(inner_type)
                } else {
                    None
                },
                var_name: name.to_token_stream(),
            }
        }
        fn get_signature(&self) -> TokenStream {
            match self.inner_type {
                Some(t) => {
                    let name = &self.var_name;
                    // Special case for Slicing.
                    let nt = if format!("{}", t) == "Slice" {
                        Ident::new("PyBindingSliceElement", t.span())
                    } else {
                        get_wrapper_ident(t)
                    };
                    if self.is_vector {
                        quote!(#name: Vec<pyo3::PyRef<#nt>>)
                    } else {
                        quote!(#name: &#nt)
                    }
                }
                None => {
                    let name = &self.var_name;
                    let t = self.initial_type;
                    quote!(#name: #t)
                }
            }
        }
        fn as_inner_argument(&self) -> TokenStream {
            match self.inner_type {
                Some(_) => {
                    let name = &self.var_name;
                    if self.is_vector {
                        quote!(#name.into_iter().map(|x| x.inner.clone()).collect())
                    } else {
                        quote!(#name.inner.clone())
                    }
                }
                None => {
                    let name = &self.var_name;
                    quote!(#name)
                }
            }
        }
    }

    struct Input {
        sig_inputs: Vec<TokenStream>,
        inner_inputs: Vec<TokenStream>,
        attr_sig: Vec<String>,
        has_receiver: bool,
    }

    impl Input {
        fn new(inputs: &Punctuated<FnArg, Comma>) -> Self {
            let mut sig = vec![];
            let mut inner = vec![];
            let mut attr_sig = vec![];
            let mut has_receiver = false;
            for arg in inputs {
                match arg {
                    FnArg::Typed(t) => {
                        let processed_argument = InputArgument::new(t);
                        sig.push(processed_argument.get_signature());
                        inner.push(processed_argument.as_inner_argument());
                        attr_sig.push(processed_argument.var_name.to_string());
                    }
                    FnArg::Receiver(slf) => {
                        sig.push(slf.into_token_stream());
                        attr_sig.push("$self".to_string());
                        has_receiver = true;
                    }
                }
            }
            attr_sig.push("/".to_string());
            Input {
                sig_inputs: sig,
                inner_inputs: inner,
                attr_sig,
                has_receiver,
            }
        }
        fn get_inner_inputs(&self) -> TokenStream {
            let inputs = &self.inner_inputs;
            quote!(#(#inputs),*)
        }
        fn get_sig_inputs(&self) -> TokenStream {
            let inputs = &self.sig_inputs;
            quote!(#(#inputs),*)
        }
        fn gen_attr_signature(&self) -> TokenStream {
            let val = vec!["(", self.attr_sig.join(", ").as_str(), ")"].join(" ");
            quote!(#[pyo3(text_signature = #val)])
        }
        fn mb_gen_staticmethod(&self, ignore: bool) -> TokenStream {
            if self.has_receiver || ignore {
                TokenStream::new()
            } else {
                quote!(#[staticmethod])
            }
        }
    }

    fn gen_attributes(attrs: &[syn::Attribute]) -> Vec<&syn::Attribute> {
        let mut result = vec![];
        let mut stop_adding_docs = false;
        for attr in attrs {
            if attr.path.is_ident("cfg") {
                result.push(attr);
            }
            if attr.path.is_ident("doc") && !stop_adding_docs {
                if format!("{}", attr.tokens).contains("# Example")
                    || format!("{}", attr.tokens).contains("# Rust crates")
                {
                    stop_adding_docs = true;
                } else {
                    result.push(attr);
                }
            }
        }
        result
    }

    fn check_in_allowlist(name: String) -> Option<TokenStream> {
        match name.as_str() {
            "create_named_tuple" => Some(quote!(
                pub fn create_named_tuple(
                    &self,
                    elements: Vec<(String, pyo3::PyRef<PyBindingNode>)>,
                ) -> pyo3::PyResult<PyBindingNode> {
                    Ok(PyBindingNode {
                        inner: self.inner.create_named_tuple(
                            elements
                                .into_iter()
                                .map(|x| (x.0, x.1.inner.clone()))
                                .collect(),
                        )?,
                    })
                }
            )),
            "constant" => Some(quote!(
                pub fn constant(&self, tv: &PyBindingTypedValue) -> pyo3::PyResult<PyBindingNode> {
                    Ok(PyBindingNode {
                        inner: self
                            .inner
                            .constant(tv.inner.t.clone(), tv.inner.value.clone())?,
                    })
                }
            )),
            "get_operation" => Some(quote!(
                pub fn get_operation(&self) -> pyo3::PyResult<String> {
                    serde_json::to_string(&self.inner.get_operation())
                        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))
                }
            )),
            "named_tuple_type" => Some(quote!(
                pub fn py_binding_named_tuple_type(
                    v: Vec<(String, pyo3::PyRef<PyBindingType>)>,
                ) -> PyBindingType {
                    PyBindingType {
                        inner: named_tuple_type(
                            v.into_iter().map(|x| (x.0, x.1.inner.clone())).collect(),
                        ),
                    }
                }
            )),
            "get_sub_values" => Some(quote!(
                fn get_sub_values(&self) -> Option<Vec<PyBindingValue>> {
                    match self.inner.get_sub_values() {
                        None => None,
                        Some(v) => {
                            Some(v.into_iter().map(|x| PyBindingValue { inner: x }).collect())
                        }
                    }
                }
            )),
            _ => None,
        }
    }
}
