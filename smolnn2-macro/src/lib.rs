use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse::Parse, parse_macro_input, punctuated::Punctuated, Expr, Ident, Token, Type, Visibility};

/// ```
/// smolnn2::model! {
///     pub MyModel: 1, 22 * 22 => 1, 1
///
///     => smolnn2::fcnn::Fcnn<22 * 22, 16>
///     => smolnn2::activation::Relu
///     => smolnn2::fcnn::Fcnn<16, 1>
///     => smolnn2::activation::Relu
/// };
/// ```
#[proc_macro]
pub fn model(input: TokenStream) -> TokenStream {
    let model = parse_macro_input!(input as Model);

    if model.layers.is_empty() {
        return quote! { compiler_error!("expected at least 1 layer in a model") }.into();
    }

    let Model { vis, name, input_w, input_h, output_w, output_h, layers, .. } = model;

    let mut layer_tokens = quote! {};
    let mut forward = quote! {};
    let mut collectors = quote! {};
    let mut back_propagate = quote! {};

    for (i, l) in layers.iter().enumerate() {
        let name = format_ident!("l{}", i + 1);
        layer_tokens.extend(quote! {
            #name: #l,
        });

        let prev = format_ident!("l{i}");
        forward.extend(quote! {
            let #name = self.#name.forward(&#prev);
        });

        let collector = format_ident!("c{}", i + 1);
        collectors.extend(quote! {
            #collector: &mut <#l as ::smolnn2::Collectable>::Collector,
        });

        let d_next = format_ident!("d{}", i + 2);
        back_propagate.extend(quote! {
            self.#name.back_propagate(#collector, #d_next, &#prev);
        });
    }

    let mut derivatives = quote! {};
    for i in (1..layers.len()).rev() {
        let d_this = format_ident!("d{}", i + 1);
        let d_last = format_ident!("d{}", i + 2);
        let layer = format_ident!("l{}", i + 1);

        derivatives.extend(quote! {
            let #d_this = self.#layer.derivative(&#d_last);
        });
    }

    let d_max = format_ident!("d{}", layers.len() + 1);
    let last_layer = format_ident!("l{}", layers.len());
    quote! {
        #[derive(Debug, Clone)]
        #vis struct #name {
            #layer_tokens
        }

        impl #name {
            pub fn forward(
                &self,
                input: &::smolmatrix::Matrix<#input_w, #input_h>,
            ) -> ::smolmatrix::Matrix<#output_w, #output_h> {
                let l0 = input;
                #forward
                #last_layer
            }

            pub fn back_propagate(
                &self,
                input: &::smolmatrix::Matrix<#input_w, #input_h>,
                expected: &::smolmatrix::Matrix<#output_w, #output_h>,
                #collectors
            ) -> ::smolmatrix::Matrix<#output_w, #output_h> {
                let l0 = input;
                #forward

                let #d_max = squared_error_derivative(#last_layer.clone(), expected);
                #derivatives

                #back_propagate

                #last_layer
            }
        }
    }.into()
}

struct Model {
    vis: Visibility,
    name: Ident,
    #[allow(unused)]
    colon: Token![:],
    input_w: Expr,
    #[allow(unused)]
    comma_1: Token![,],
    input_h: Expr,
    #[allow(unused)]
    arrow: Token![=>],
    output_w: Expr,
    #[allow(unused)]
    comma_2: Token![,],
    output_h: Expr,
    #[allow(unused)]
    fat_arrow: Token![=>],
    layers: Punctuated<Layer, Token![=>]>,
}

impl Parse for Model {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(Self {
            vis: input.parse()?,
            name: input.parse()?,
            colon: input.parse()?,
            input_w: input.parse()?,
            comma_1: input.parse()?,
            input_h: input.parse()?,
            arrow: input.parse()?,
            output_w: input.parse()?,
            comma_2: input.parse()?,
            output_h: input.parse()?,
            fat_arrow: input.parse()?,
            layers: input.parse_terminated(Layer::parse, Token![=>])?,
        })
    }
}

type Layer = Type;
