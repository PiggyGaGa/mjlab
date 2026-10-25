mjlab.scene
===========

.. automodule:: mjlab.scene

.. rubric:: Classes

.. hlist::
  :columns: 3

  - :class:`Scene`
  - :class:`SceneCfg`
  - :class:`GaussianSplatAssetCfg`
  - :class:`GaussianSplatSceneCfg`
  - :class:`SceneManifest`

Scene
-----

.. autoclass:: Scene
  :members:

.. autoclass:: SceneCfg
  :members:
  :exclude-members: __init__
  :undoc-members:

3D Gaussian Splatting
---------------------

.. autoclass:: GaussianSplatAssetCfg
  :members:
  :exclude-members: __init__

.. autoclass:: GaussianSplatSceneCfg
  :members:
  :exclude-members: __init__

Scene manifests
---------------

.. autoclass:: SceneManifest
  :members:

.. autoclass:: SceneManifestAsset
  :members:

.. autoclass:: SceneManifestFile
  :members:

.. autoclass:: SceneSpawnPoint
  :members:

.. autofunction:: load_scene_manifest

.. autofunction:: apply_scene_manifest
