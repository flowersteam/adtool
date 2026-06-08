import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { CAMERA_DEPTH_BOUNDS } from "./config.js";
import { clamp } from "./utils.js";

export function createMapScene(container) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#eef0ec");

    const camera = new THREE.PerspectiveCamera(54, 1, 0.01, 800);
    camera.position.set(0, 0, 26);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enableRotate = false;
    controls.screenSpacePanning = true;
    controls.zoomSpeed = 1.05;
    controls.panSpeed = 0.85;
    controls.maxDistance = CAMERA_DEPTH_BOUNDS.max;
    controls.minDistance = CAMERA_DEPTH_BOUNDS.min;

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const textureLoader = new THREE.TextureLoader();
    let animationStarted = false;

    function resizeRenderer() {
        const width = Math.max(1, container.clientWidth);
        const height = Math.max(1, container.clientHeight);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height, false);
    }

    function fitView(planes) {
        if (planes.length === 0) {
            camera.position.set(0, 0, 26);
            controls.target.set(0, 0, 0);
            controls.update();
            return;
        }

        const box = new THREE.Box3();
        for (const plane of planes) {
            box.expandByPoint(plane.position);
        }

        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxSpan = Math.max(size.x, size.y, 1);
        const fov = camera.fov * (Math.PI / 180);
        const distance = clamp(
            (maxSpan / 2) / Math.tan(fov / 2) + 5,
            CAMERA_DEPTH_BOUNDS.min,
            CAMERA_DEPTH_BOUNDS.max,
        );

        controls.target.set(center.x, center.y, 0);
        camera.position.set(center.x, center.y, distance);
        controls.update();
    }

    function setPointerFromEvent(event) {
        const rect = renderer.domElement.getBoundingClientRect();
        pointer.x = ((event.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
        pointer.y = -((event.clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
    }

    function pickPlaneAtPointer(event, planes) {
        setPointerFromEvent(event);
        raycaster.setFromCamera(pointer, camera);
        const intersects = raycaster.intersectObjects(planes);
        return intersects.length > 0 ? intersects[0].object : null;
    }

    function updateViewAnimation(getPlanes) {
        requestAnimationFrame(() => updateViewAnimation(getPlanes));
        controls.update();

        for (const plane of getPlanes()) {
            const distance = Math.max(0.01, camera.position.z - plane.position.z);
            const scale = distance * 0.19 * (plane.userData.scaleBoost || 1.0);
            plane.scale.set(scale, scale, 1);
        }

        renderer.render(scene, camera);
    }

    function startAnimation(getPlanes) {
        if (animationStarted) {
            return;
        }
        animationStarted = true;
        updateViewAnimation(getPlanes);
    }

    return {
        fitView,
        pickPlaneAtPointer,
        renderer,
        resizeRenderer,
        scene,
        startAnimation,
        textureLoader,
    };
}
