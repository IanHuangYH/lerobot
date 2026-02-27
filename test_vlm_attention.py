#!/usr/bin/env python
"""Quick test script to verify VLM attention saving works correctly."""

import torch
from pathlib import Path

# Track test results
all_tests_passed = True

# Test 1: Check if VLM attention methods exist in model
print("=" * 80)
print("Test 1: Check VLM attention methods in PI05Pytorch")
print("=" * 80)

try:
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    
    # Check methods exist
    assert hasattr(PI05Pytorch, 'enable_vlm_attention_map_saving'), "Missing: enable_vlm_attention_map_saving"
    assert hasattr(PI05Pytorch, 'disable_vlm_attention_map_saving'), "Missing: disable_vlm_attention_map_saving"
    assert hasattr(PI05Pytorch, 'get_vlm_attention_maps'), "Missing: get_vlm_attention_maps"
    assert hasattr(PI05Pytorch, 'clear_vlm_attention_maps'), "Missing: clear_vlm_attention_maps"
    
    print("✓ All VLM attention methods exist in PI05Pytorch")
except Exception as e:
    print(f"✗ Error: {e}")
    all_tests_passed = False

# Test 2: Check if config flag exists
print("\n" + "=" * 80)
print("Test 2: Check config flag in EvalConfig")
print("=" * 80)

try:
    from lerobot.configs.default import EvalConfig
    
    config = EvalConfig()
    assert hasattr(config, 'save_vlm_attention_maps'), "Missing: save_vlm_attention_maps in EvalConfig"
    assert isinstance(config.save_vlm_attention_maps, bool), "save_vlm_attention_maps should be bool"
    assert config.save_vlm_attention_maps == False, "Default should be False"
    
    print(f"✓ EvalConfig has save_vlm_attention_maps flag (default: {config.save_vlm_attention_maps})")
except Exception as e:
    print(f"✗ Error: {e}")
    all_tests_passed = False

# Test 3: Check if helper function exists
print("\n" + "=" * 80)
print("Test 3: Check extraction helper function")
print("=" * 80)

try:
    from lerobot.scripts.lerobot_eval import _extract_batch_sample_from_vlm_attention_maps
    
    # Test with mock data
    mock_vlm_attention = [{
        'rollout_step': 0,
        'vlm_attention': {
            'prefix_len': 968,
            'attention_weights': {
                17: torch.randn(2, 8, 968, 968)  # batch=2
            }
        }
    }]
    
    # Extract batch 0
    extracted = _extract_batch_sample_from_vlm_attention_maps(mock_vlm_attention, batch_idx=0)
    
    assert len(extracted) == 1, "Should have 1 rollout step"
    assert extracted[0]['rollout_step'] == 0
    assert extracted[0]['vlm_attention']['prefix_len'] == 968
    assert 17 in extracted[0]['vlm_attention']['attention_weights']
    assert extracted[0]['vlm_attention']['attention_weights'][17].shape[0] == 1, "Batch size should be 1"
    
    # Verify redundant fields are removed
    assert 'num_img_tokens' not in extracted[0]['vlm_attention'], "num_img_tokens should be removed"
    assert 'num_cameras' not in extracted[0]['vlm_attention'], "num_cameras should be removed"
    
    print("✓ VLM attention extraction helper works correctly")
    print(f"  - Input shape: torch.Size([2, 8, 968, 968])")
    print(f"  - Output shape: {extracted[0]['vlm_attention']['attention_weights'][17].shape}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    all_tests_passed = False

# Test 4: Verify file structure expectations
print("\n" + "=" * 80)
print("Test 4: Check if existing attention files exist (optional)")
print("=" * 80)

attention_dir = Path("eval_logs/quick_test/attention/libero_object_0")
if attention_dir.exists():
    files = list(attention_dir.glob("episode_*_attention.pt"))
    if files:
        print(f"✓ Found {len(files)} existing action attention files")
        # Load one to verify structure
        data = torch.load(files[0], map_location='cpu')
        print(f"  - Rollout steps: {len(data['rollout_steps'])}")
        print(f"  - Denoising steps: {data['metadata']['num_denoising_steps_per_action']}")
        
        # Show expected VLM attention structure
        print("\n  Expected VLM attention file structure:")
        print("  {")
        print("    'episode_index': int,")
        print("    'batch_index': int,")
        print("    'rollout_steps': [")
        print("      {")
        print("        'rollout_step': int,")
        print("        'vlm_attention': {")
        print("          'prefix_len': int,  # 768 (images) + 200 (language) = 968")
        print("          'attention_weights': {layer_idx: tensor}  # [1, 8, 968, 968]")
        print("        }")
        print("      },")
        print("      ...")
        print("    ],")
        print("    'metadata': {'num_rollout_steps': int}")
        print("  }")
        print("\n  Note: num_img_tokens=256 and num_cameras can be derived from prefix_len")
    else:
        print("  No existing attention files found (run evaluation first)")
else:
    print("  No attention directory found (run evaluation first)")

# Test 5: Verify VLM attention capture logic
print("\n" + "=" * 80)
print("Test 5: Verify VLM attention capture from forward pass")
print("=" * 80)

try:
    from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel, get_gemma_config
    
    # Create minimal model configuration
    vlm_config = get_gemma_config("gemma_2b")
    action_expert_config = get_gemma_config("gemma_300m")
    
    print("  Creating minimal PaliGemma model...")
    model = PaliGemmaWithExpertModel(
        vlm_config=vlm_config,
        action_expert_config=action_expert_config,
        use_adarms=[False, True],
        precision="float32",
        image_size=224,
        freeze_vision_encoder=False,
        train_expert_only=False,
    )
    
    # Create dummy inputs for prefix-only forward pass
    batch_size = 1
    prefix_len = 968  # 768 image + 200 language
    hidden_dim = 2048
    
    print("  Creating dummy prefix embeddings...")
    prefix_embs = torch.randn(batch_size, prefix_len, hidden_dim)
    attention_mask = torch.zeros(batch_size, 1, prefix_len, prefix_len)
    position_ids = torch.arange(prefix_len).unsqueeze(0)
    
    # CRITICAL: Set attention implementation to 'eager' to support output_attentions
    model.paligemma.language_model.config._attn_implementation = "eager"
    
    print("  Testing prefix-only forward WITHOUT attention capture...")
    # Test without attention capture
    result_no_att = model.forward(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],  # Prefix only
        use_cache=True,
        adarms_cond=[None, None],
        return_attention_weights=False,
    )
    assert len(result_no_att) == 2, "Should return (outputs, past_key_values)"
    print("  ✓ Forward pass without attention works")
    
    print("  Testing prefix-only forward WITH attention capture...")
    # Test with attention capture
    result_with_att = model.forward(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],  # Prefix only
        use_cache=True,
        adarms_cond=[None, None],
        return_attention_weights=True,
    )
    assert len(result_with_att) == 3, "Should return (outputs, past_key_values, attention_weights)"
    
    outputs, past_kv, att_weights = result_with_att
    assert att_weights is not None, "Attention weights should not be None"
    assert isinstance(att_weights, dict), "Attention weights should be a dict"
    assert len(att_weights) > 0, "Should have attention from at least one layer"
    
    # Check attention shape
    first_layer_att = list(att_weights.values())[0]
    expected_shape = (batch_size, 8, prefix_len, prefix_len)  # (B, heads, seq, seq)
    assert first_layer_att.shape == expected_shape, f"Expected shape {expected_shape}, got {first_layer_att.shape}"
    
    print(f"  ✓ VLM attention captured successfully!")
    print(f"    - Layers captured: {len(att_weights)}")
    print(f"    - Attention shape per layer: {first_layer_att.shape}")
    print(f"    - Total elements: {first_layer_att.numel():,}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    all_tests_passed = False

print("\n" + "=" * 80)
if all_tests_passed:
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nVLM attention saving implementation is ready.")
    print("\nTo test it:")
    print("1. Edit eval_libero_quick_test.sh: change --eval.save_vlm_attention_maps=false to true")
    print("2. Run: bash pi_setting/eval/eval_libero_quick_test.sh")
    print("3. Check: eval_logs/quick_test/vlm_attention/libero_object_0/episode_00000_vlm_attention.pt")
    exit(0)
else:
    print("✗ SOME TESTS FAILED!")
    print("=" * 80)
    print("\nPlease check the errors above and fix the issues.")
    exit(1)
