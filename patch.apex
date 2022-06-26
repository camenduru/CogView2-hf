diff --git a/setup.py b/setup.py
index 5f68ecf..b4d44a8 100644
--- a/setup.py
+++ b/setup.py
@@ -30,15 +30,15 @@ def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
     print("\nCompiling cuda extensions with")
     print(raw_output + "from " + cuda_dir + "/bin\n")
 
-    if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
-        raise RuntimeError(
-            "Cuda extensions are being compiled with a version of Cuda that does "
-            "not match the version used to compile Pytorch binaries.  "
-            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
-            + "In some cases, a minor-version mismatch will not cause later errors:  "
-            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
-            "You can try commenting out this check (at your own risk)."
-        )
+#    if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
+#        raise RuntimeError(
+#            "Cuda extensions are being compiled with a version of Cuda that does "
+#            "not match the version used to compile Pytorch binaries.  "
+#            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
+#            + "In some cases, a minor-version mismatch will not cause later errors:  "
+#            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
+#            "You can try commenting out this check (at your own risk)."
+#        )
 
 
 def raise_if_cuda_home_none(global_option: str) -> None:
