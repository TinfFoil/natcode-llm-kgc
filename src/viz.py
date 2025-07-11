def get_attn(self, model, tokenizer, sample, icl_prompt):
    text = sample['text']
    if self.config['natlang']:
        prompt = self.make_natlang_prompt(ICL_prompt=icl_prompt, sample_text=text, triples=[])
    else:
        prompt = self.make_code_prompt(ICL_prompt=icl_prompt, sample_text=text, triples=[])
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(inputs.input_ids,
                            num_return_sequences=1,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.eos_token_id,
                            max_new_tokens = 1000,
                            return_dict_in_generate=True,
                            output_attentions = True,
                            )
    attn_list = []
    for i, tokens in enumerate(outputs.attentions[1:]):
        # for layer in tokens:
        attn = torch.mean(tokens[-1].squeeze(), dim = 0)
        attn_list.append(attn)

    input_len = inputs.input_ids.shape[1]
    
    output_no_prompt = outputs.sequences.squeeze()[input_len:]

    result = tokenizer.decode(output_no_prompt,
                            skip_special_tokens=True,
                            )
    true = sample['triple_list']        
    pred = self.extract_triples(result)
    print('pred', result)
    stats = self.calculate_strict_micro_f1([true], [pred])
    print('results:', stats)
    print('gt', true)

    return attn_list, outputs.sequences.squeeze(), stats

def heatmap2d(
    self,
    attention_scores,
    token_ids_full,
    idx,
    width=10,         # number of columns in the grid
    height=None,      # maximum number of rows (if provided)
    cell_w=3,       # width of each cell (in figure units)
    cell_h=0.75,       # height of each cell (in figure units)
    cmap='viridis',
    norm='log',
    high_threshold=0.999,
    savename='',
    format='pdf',
    text_mode='truncate',   # 'wrap' or 'truncate'
    wrap_width=10,      # maximum characters per line if wrapping
    truncate_length=5  # maximum total characters if truncating
):
    """
    Plots a 2D heatmap of attention scores with token labels in each cell.
    Allows customizing the cell's width and height separately (rather than only squares).
    Also supports wrapping or truncating text within the cells.

    Args:
        attention_scores: 1D array (num_tokens,) of attention values.
        token_ids_full:   The list/array of token IDs.
        idx:              Index or identifier used in the filename.
        width:            Number of columns in the grid.
        height:           Maximum number of rows in the grid (if None, computed automatically).
        cell_w:           Width of each cell in figure coordinates.
        cell_h:           Height of each cell in figure coordinates.
        cmap:             Matplotlib colormap.
        norm:             Normalization type, 'log' or 'linear'.
        high_threshold:   Quantile above which cells will be colored black.
        savename:         Directory or prefix to save the output file.
        format:           Output format for saving the figure (e.g., 'pdf', 'png').
        text_mode:        Either 'wrap' (to wrap text) or 'truncate' (to truncate text).
        wrap_width:       Maximum characters per line if text_mode is 'wrap'.
        truncate_length:  Maximum total characters if text_mode is 'truncate'.
    """
    attention_scores = attention_scores.float().cpu().numpy()
    num_tokens = attention_scores.shape[0]
    next_token = token_ids_full[:num_tokens][-1]
    token_ids = token_ids_full[:num_tokens]
    heatmap_token = self.tokenizer.decode(next_token)
    print('heatmap token:', heatmap_token)

    labels = [self.tokenizer.decode(token) for token in token_ids]
    assert len(attention_scores) == len(labels)
    
    # Calculate the number of rows and columns
    n_cols = width
    n_rows = int(np.ceil(num_tokens / n_cols))
    if height:
        n_rows = min(n_rows, height)
    
    # Pad the attention scores and labels if necessary
    pad_length = n_rows * n_cols - num_tokens
    attention_scores_padded = np.pad(attention_scores, (0, pad_length), mode='constant', constant_values=np.nan)
    labels_padded = labels + [''] * pad_length
    
    # Reshape the data into a 2D grid
    attention_scores_2d = attention_scores_padded.reshape(n_rows, n_cols)
    labels_2d = np.array(labels_padded).reshape(n_rows, n_cols)
    
    # Create the figure and axis using cell dimensions
    fig_width = n_cols * cell_w
    fig_height = n_rows * cell_h
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Choose the normalization
    if norm == 'log':
        color_norm = LogNorm(vmin=np.nanmin(attention_scores), vmax=np.nanmax(attention_scores))
    else:
        color_norm = Normalize(vmin=np.nanmin(attention_scores), vmax=np.nanmax(attention_scores))
    
    # Create the heatmap with non-square cells
    im = ax.imshow(attention_scores_2d, cmap=cmap, aspect='auto', norm=color_norm)
    
    # Apply black color to high values
    high_mask = attention_scores_2d > np.nanquantile(attention_scores, high_threshold)
    im.cmap.set_bad(color='black')
    attention_scores_2d_masked = np.ma.masked_where(high_mask, attention_scores_2d)
    im.set_data(attention_scores_2d_masked)
    
    # Helper function to process text based on mode
    def process_text(text):
        if text_mode == 'wrap':
            return "\n".join(textwrap.wrap(text, width=wrap_width))
        elif text_mode == 'truncate':
            return text[:truncate_length] + "..." if len(text) > truncate_length else text
        else:
            return text
    
    # Add labels to each cell
    for i in range(n_rows):
        for j in range(n_cols):
            text = labels_2d[i, j]
            score = attention_scores_2d[i, j]
            if not np.isnan(score):
                processed_text = process_text(text)
                ax.text(j, i, processed_text, ha='center', va='center', fontsize=30, color='white', fontweight='bold')
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, aspect=20)
    cbar.ax.tick_params(labelsize=26)
    
    plt.tight_layout()
    save_dir = f"./paper/attn/{savename}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'attn_{idx}_{heatmap_token}.{format}'), format=format, bbox_inches='tight')
    plt.close()

