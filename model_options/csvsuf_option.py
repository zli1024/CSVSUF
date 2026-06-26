
def merge_csvsuf_opt(parser):
    parser.add_argument("--ANE", type=bool, default=False, help="whether use adaptive noise estimator")
    parser.add_argument("--MLP_Cps", type=bool, default=False, help="whether use x-update loss compensation")
    parser.add_argument("--in_ch", type=int, default=16, help="Fusion4D Transformer input channel")
    parser.add_argument("--out_ch", type=int, default=17, help="Fusion4D Transformer output channel")
    parser.add_argument("--embed_dim", type=int, default=16, help="Fusion4D Transformer output channel")
    parser.add_argument("--depths", type=list, default=[3, 3, 3], help="Fusion4D Transformer depths")
    parser.add_argument("--num_heads", type=list, default=[2, 2, 2], help="Fusion4D Transformer number of heads")
    parser.add_argument("--window_size", type=list, default=[5, 8, 8], help="Fusion4D Transformer number of heads")
    parser.add_argument("--qkv_bias", type=bool, default=True, help="Fusion4D Transformer whether use qkv bias")
    parser.add_argument("--drop_path_rate", type=float, default=0.2, help="Drop path rate")
    parser.add_argument("--use_checkpoint_attn", type=bool, default=True, help="whether use checkpoint for attention blocks")
    parser.add_argument("--use_checkpoint_ffn", type=bool, default=True, help="whether use checkpoint for ffn modules")
    parser.add_argument("--no_checkpoint_attn_blocks", type=list, default=[], help="which attn block dont use checkpoint")
    parser.add_argument("--no_checkpoint_ffn_blocks", type=list, default=[], help="which ffn block dont use checkpoint")
    
    return parser